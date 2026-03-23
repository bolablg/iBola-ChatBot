"""
Advanced Conversation Memory Service with summarization and compression.
"""

import hashlib
import json
import os
import sys
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import GEMINI_API_KEY

try:
    from langchain_classic.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("Advanced memory service requires langchain-google-genai")


class AdvancedMemoryManager:
    """
    Advanced conversation memory with summarization, compression, and context management.
    """

    def __init__(self, max_memory_items: int = 100, summary_interval: int = 10):
        self.max_memory_items = max_memory_items
        self.summary_interval = summary_interval
        self.memory_store = {}  # {session_id: MemorySession}
        self.compression_cache = {}  # Cache for compressed memories

        if MEMORY_AVAILABLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY
            )

            # Initialize summarization chain
            self.summary_prompt = PromptTemplate.from_template("""
            Summarize the following conversation, focusing on:
            1. Key topics discussed
            2. User's main interests and questions
            3. Important facts or preferences revealed
            4. Any unresolved questions or follow-ups needed

            Keep the summary concise but comprehensive.

            Conversation:
            {conversation}

            Summary:""")

            self.summary_chain = LLMChain(
                llm=self.llm, prompt=self.summary_prompt, verbose=False
            )

            # Compression chain for old memories
            self.compression_prompt = PromptTemplate.from_template("""
            Compress the following conversation summary into key facts and insights.
            Focus on the most important information that would be relevant for future conversations.
            Make it as concise as possible while preserving essential details.

            Original Summary:
            {summary}

            Compressed Version:""")

            self.compression_chain = LLMChain(
                llm=self.llm, prompt=self.compression_prompt, verbose=False
            )

    def get_memory(self, session_id: str) -> "MemorySession":
        """Get or create memory session for user."""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = MemorySession(
                session_id, self.max_memory_items
            )
        return self.memory_store[session_id]

    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        agent_type: str,
        metadata: Dict[str, Any] = None,
    ):
        """Add a new interaction to memory."""
        memory_session = self.get_memory(session_id)

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "agent_response": agent_response,
            "agent_type": agent_type,
            "metadata": metadata or {},
        }

        memory_session.add_interaction(interaction)

        # Trigger summarization if needed
        if len(memory_session.interactions) % self.summary_interval == 0:
            self._summarize_recent_interactions(session_id)

        # Compress old summaries if memory is getting full
        if len(memory_session.interactions) > self.max_memory_items * 0.8:
            self._compress_old_memories(session_id)

    def get_context(self, session_id: str, max_items: int = 5) -> List[Dict]:
        """Get relevant context for current conversation."""
        memory_session = self.get_memory(session_id)

        # Get recent interactions
        recent = memory_session.get_recent_interactions(max_items)

        # Add relevant summaries if available
        context = []
        if memory_session.summaries:
            # Get the most recent summary
            recent_summary = memory_session.summaries[-1]
            context.append(
                {
                    "type": "summary",
                    "content": recent_summary["summary"],
                    "timestamp": recent_summary["timestamp"],
                }
            )

        # Add recent interactions
        context.extend(recent)

        return context

    def get_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """Get memory statistics for session."""
        memory_session = self.get_memory(session_id)

        return {
            "total_interactions": len(memory_session.interactions),
            "total_summaries": len(memory_session.summaries),
            "total_compressed": len(memory_session.compressed_memories),
            "last_interaction": (
                memory_session.interactions[-1]["timestamp"]
                if memory_session.interactions
                else None
            ),
            "memory_usage": len(memory_session.interactions) / self.max_memory_items,
        }

    def _summarize_recent_interactions(self, session_id: str):
        """Summarize recent interactions."""
        if not MEMORY_AVAILABLE:
            return

        memory_session = self.get_memory(session_id)

        # Get last N interactions to summarize
        recent_interactions = memory_session.get_recent_interactions(
            self.summary_interval
        )

        if len(recent_interactions) < 3:  # Don't summarize if too few interactions
            return

        # Format conversation for summarization
        conversation_text = "\n".join(
            [
                f"User: {interaction['user_message']}\nAssistant: {interaction['agent_response']}"
                for interaction in recent_interactions
            ]
        )

        try:
            summary_result = self.summary_chain.run(conversation=conversation_text)

            summary = {
                "timestamp": datetime.now().isoformat(),
                "interactions_covered": len(recent_interactions),
                "summary": summary_result.strip(),
                "interaction_range": {
                    "start": recent_interactions[0]["timestamp"],
                    "end": recent_interactions[-1]["timestamp"],
                },
            }

            memory_session.add_summary(summary)
            print(
                f"📝 Summarized {len(recent_interactions)} interactions for session {session_id}"
            )

        except Exception as e:
            print(f"❌ Error summarizing interactions: {e}")

    def _compress_old_memories(self, session_id: str):
        """Compress old memories to save space."""
        if not MEMORY_AVAILABLE:
            return

        memory_session = self.get_memory(session_id)

        # Compress oldest summaries if we have more than 5
        if len(memory_session.summaries) > 5:
            oldest_summaries = memory_session.summaries[
                :-3
            ]  # Keep last 3 summaries uncompressed

            for summary in oldest_summaries:
                cache_key = hashlib.md5(
                    summary["summary"].encode(), usedforsecurity=False
                ).hexdigest()

                if cache_key not in self.compression_cache:
                    try:
                        compressed = self.compression_chain.run(
                            summary=summary["summary"]
                        )
                        self.compression_cache[cache_key] = compressed.strip()
                    except Exception as e:
                        print(f"❌ Error compressing memory: {e}")
                        continue

                # Replace summary with compressed version
                summary["original_summary"] = summary["summary"]
                summary["summary"] = self.compression_cache[cache_key]
                summary["compressed"] = True

            print(
                f"🗜️ Compressed {len(oldest_summaries)} old summaries for session {session_id}"
            )

    def search_memory(self, session_id: str, query: str, limit: int = 3) -> List[Dict]:
        """Search through conversation memory for relevant information."""
        if not MEMORY_AVAILABLE:
            return []

        memory_session = self.get_memory(session_id)

        # Search through interactions
        relevant_interactions = []
        query_lower = query.lower()

        for interaction in reversed(
            memory_session.interactions
        ):  # Search most recent first
            if (
                query_lower in interaction["user_message"].lower()
                or query_lower in interaction["agent_response"].lower()
            ):
                relevant_interactions.append(interaction)
                if len(relevant_interactions) >= limit:
                    break

        return relevant_interactions

    def clear_memory(self, session_id: str):
        """Clear all memory for a session."""
        if session_id in self.memory_store:
            del self.memory_store[session_id]
            print(f"🗑️ Cleared memory for session {session_id}")


class MemorySession:
    """Represents a single user's conversation memory."""

    def __init__(self, session_id: str, max_items: int):
        self.session_id = session_id
        self.max_items = max_items
        self.interactions = deque(maxlen=max_items * 2)  # Allow some overflow
        self.summaries = []
        self.compressed_memories = []

    def add_interaction(self, interaction: Dict):
        """Add an interaction to memory."""
        self.interactions.append(interaction)

        # Clean up old interactions if we exceed limit
        while len(self.interactions) > self.max_items:
            self.interactions.popleft()

    def add_summary(self, summary: Dict):
        """Add a summary to memory."""
        self.summaries.append(summary)

        # Keep only last 10 summaries
        if len(self.summaries) > 10:
            self.summaries.pop(0)

    def get_recent_interactions(self, count: int) -> List[Dict]:
        """Get the most recent interactions."""
        return list(self.interactions)[-count:]

    def get_interactions_by_agent(self, agent_type: str) -> List[Dict]:
        """Get all interactions with a specific agent type."""
        return [
            interaction
            for interaction in self.interactions
            if interaction["agent_type"] == agent_type
        ]


# Global memory manager instance
advanced_memory = AdvancedMemoryManager() if MEMORY_AVAILABLE else None
