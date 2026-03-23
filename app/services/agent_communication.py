"""
Inter-agent Communication and Task Delegation System.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import GEMINI_API_KEY

try:
    from langchain_classic.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    COMMUNICATION_AVAILABLE = True
except ImportError:
    COMMUNICATION_AVAILABLE = False
    print("Agent communication service requires langchain-google-genai")


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a task that can be delegated between agents."""

    task_id: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_by: str
    assigned_to: Optional[str]
    created_at: datetime
    deadline: Optional[datetime]
    dependencies: List[str]  # Task IDs this task depends on
    subtasks: List[str]  # Subtask IDs
    metadata: Dict[str, Any]
    result: Optional[Any]


class AgentMessage:
    """Represents a message between agents."""

    def __init__(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: Any,
        metadata: Dict[str, Any] = None,
    ):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.message_id = f"{sender}_{recipient}_{int(self.timestamp.timestamp())}"


class AgentCommunicationHub:
    """
    Central hub for inter-agent communication and task delegation.
    """

    def __init__(self):
        self.agents = {}  # {agent_name: agent_instance}
        self.tasks = {}  # {task_id: Task}
        self.messages = []  # List of AgentMessage objects
        self.task_queues = {}  # {agent_name: List[Task]}
        self.capabilities = {}  # {agent_name: List[str]}

        if COMMUNICATION_AVAILABLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", temperature=0.2, google_api_key=GEMINI_API_KEY
            )

            # Task delegation analysis chain
            self.delegation_prompt = PromptTemplate.from_template("""
            Analyze if this task should be delegated to another agent.

            Current Agent: {current_agent}
            Current Agent Capabilities: {current_capabilities}

            Available Agents and their capabilities:
            {available_agents}

            Task: {task_description}
            Context: {context}

            Should this task be delegated? If yes, which agent and why?
            Consider:
            1. Agent expertise and capabilities
            2. Task complexity and requirements
            3. Current agent workload
            4. Inter-agent dependencies

            Response format:
            Delegate: [YES/NO]
            Target_Agent: [agent_name or NONE]
            Reasoning: [brief explanation]
            Confidence: [HIGH/MEDIUM/LOW]
            """)

            self.delegation_chain = LLMChain(
                llm=self.llm, prompt=self.delegation_prompt, verbose=False
            )

    def register_agent(
        self, agent_name: str, agent_instance: Any, capabilities: List[str]
    ):
        """Register an agent with the communication hub."""
        self.agents[agent_name] = agent_instance
        self.capabilities[agent_name] = capabilities
        self.task_queues[agent_name] = []
        print(f"📝 Registered agent: {agent_name} with capabilities: {capabilities}")

    def unregister_agent(self, agent_name: str):
        """Unregister an agent from the communication hub."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.capabilities[agent_name]
            del self.task_queues[agent_name]
            print(f"🗑️ Unregistered agent: {agent_name}")

    def send_message(self, message: AgentMessage):
        """Send a message between agents."""
        self.messages.append(message)

        # If recipient exists, deliver the message
        if message.recipient in self.agents:
            recipient_agent = self.agents[message.recipient]
            if hasattr(recipient_agent, "receive_message"):
                # Use asyncio to handle async message delivery
                asyncio.create_task(
                    self._deliver_message_async(recipient_agent, message)
                )

        print(
            f"📨 Message from {message.sender} to {message.recipient}: {message.message_type}"
        )

    async def _deliver_message_async(self, recipient_agent, message):
        """Asynchronously deliver message to recipient agent."""
        try:
            await recipient_agent.receive_message(message)
        except Exception as e:
            print(f"❌ Error delivering message to {message.recipient}: {e}")

    def create_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        created_by: str = "system",
        deadline: Optional[datetime] = None,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Create a new task."""
        task_id = f"task_{int(datetime.now().timestamp())}_{len(self.tasks)}"

        task = Task(
            task_id=task_id,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            created_by=created_by,
            assigned_to=None,
            created_at=datetime.now(),
            deadline=deadline,
            dependencies=dependencies or [],
            subtasks=[],
            metadata=metadata or {},
            result=None,
        )

        self.tasks[task_id] = task
        print(f"🎯 Created task: {task_id} - {description}")
        return task_id

    def delegate_task(self, task_id: str, from_agent: str, to_agent: str) -> bool:
        """Delegate a task from one agent to another."""
        if task_id not in self.tasks:
            print(f"❌ Task {task_id} not found")
            return False

        if to_agent not in self.agents:
            print(f"❌ Target agent {to_agent} not found")
            return False

        task = self.tasks[task_id]
        task.assigned_to = to_agent
        task.status = TaskStatus.DELEGATED

        # Move task to target agent's queue
        self.task_queues[to_agent].append(task)

        # Send notification message
        message = AgentMessage(
            sender=from_agent,
            recipient=to_agent,
            message_type="task_delegation",
            content={"task_id": task_id, "task": task},
            metadata={"delegated_from": from_agent},
        )
        self.send_message(message)

        print(f"🔄 Delegated task {task_id} from {from_agent} to {to_agent}")
        return True

    def analyze_delegation(
        self, current_agent: str, task_description: str, context: str = ""
    ) -> Dict[str, Any]:
        """Analyze if a task should be delegated using AI."""
        if not COMMUNICATION_AVAILABLE:
            return {
                "delegate": False,
                "reasoning": "Communication service not available",
            }

        # Get available agents and their capabilities
        available_agents = {}
        for agent_name, capabilities in self.capabilities.items():
            if agent_name != current_agent:
                available_agents[agent_name] = capabilities

        current_caps = self.capabilities.get(current_agent, [])

        try:
            result = self.delegation_chain.run(
                current_agent=current_agent,
                current_capabilities=", ".join(current_caps),
                available_agents=json.dumps(available_agents, indent=2),
                task_description=task_description,
                context=context,
            )

            # Parse the result
            lines = result.strip().split("\n")
            delegation_info = {}

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    delegation_info[key.strip().lower()] = value.strip()

            delegate = delegation_info.get("delegate", "NO").upper() == "YES"
            target_agent = delegation_info.get("target_agent", "NONE")

            if target_agent == "NONE":
                target_agent = None

            return {
                "delegate": delegate,
                "target_agent": target_agent,
                "reasoning": delegation_info.get("reasoning", ""),
                "confidence": delegation_info.get("confidence", "LOW"),
            }

        except Exception as e:
            print(f"❌ Error analyzing delegation: {e}")
            return {"delegate": False, "reasoning": f"Analysis failed: {e}"}

    def get_agent_workload(self, agent_name: str) -> int:
        """Get the current workload of an agent."""
        if agent_name not in self.task_queues:
            return 0

        pending_tasks = [
            task
            for task in self.task_queues[agent_name]
            if task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
        ]
        return len(pending_tasks)

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """Get capabilities of a specific agent."""
        return self.capabilities.get(agent_name, [])

    def find_best_agent_for_task(
        self, task_description: str, required_capabilities: List[str] = None
    ) -> Optional[str]:
        """Find the best agent for a specific task."""
        best_agent = None
        best_score = -1

        for agent_name, capabilities in self.capabilities.items():
            score = 0

            # Check if agent has required capabilities
            if required_capabilities:
                matching_caps = len(set(capabilities) & set(required_capabilities))
                score += matching_caps * 10

            # Consider workload (lower workload is better)
            workload = self.get_agent_workload(agent_name)
            score -= workload * 2

            # Check task relevance based on keywords
            task_lower = task_description.lower()
            for capability in capabilities:
                if capability.lower() in task_lower:
                    score += 5

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def broadcast_message(
        self,
        sender: str,
        message_type: str,
        content: Any,
        metadata: Dict[str, Any] = None,
    ):
        """Broadcast a message to all registered agents."""
        for agent_name in self.agents:
            if agent_name != sender:
                message = AgentMessage(
                    sender=sender,
                    recipient=agent_name,
                    message_type=message_type,
                    content=content,
                    metadata=metadata,
                )
                self.send_message(message)

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the agent communication system."""
        return {
            "total_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "total_messages": len(self.messages),
            "agent_workloads": {
                agent: self.get_agent_workload(agent) for agent in self.agents
            },
            "capabilities": self.capabilities,
        }


# Global communication hub instance
communication_hub = AgentCommunicationHub() if COMMUNICATION_AVAILABLE else None
