"""
Orchestrator - Routes queries to the appropriate specialized agent.
"""

import re
from typing import Dict, Any, List, Tuple
from .professional_agent import ProfessionalAgent
from .education_agent import EducationAgent
from .learning_agent import LearningAgent
from .redirect_agent import RedirectAgent
from .classification_agent import ClassificationAgent
from app.services.dynamic_guardrails import dynamic_guardrails
from app.services.google_chat_alert import google_chat_alert
from app.services.language_detection import language_service

class AgentOrchestrator:
    """Routes queries to appropriate specialized agents."""

    def __init__(self):
        # Initialize all agents
        self.classification_agent = ClassificationAgent()
        self.professional_agent = ProfessionalAgent()
        self.education_agent = EducationAgent()
        self.learning_agent = LearningAgent()
        self.redirect_agent = RedirectAgent()

        # Session tracking for redirect counts and user context
        self.session_data = {}  # {session_id: {'redirect_count': int, 'language': str, 'last_agent': str}}

        # Keep dynamic guardrails as fallback
        self.guardrails = dynamic_guardrails

        # Legacy patterns (still useful for regex matching)
        self.professional_patterns = [
            r'\b(what|which|where)\s+(do|did|does|is|are|was|were)\s+(you|i|he|bolaji)\s+(work|do)\b',
            r'\b(your|his|bolaji.?s)\s+(job|role|position|career)\b',
            r'\b(experience|background|resume|cv)\b',
            r'\b(skill|technology|tool|expertise)\b',
            r'\b(project|achievement|accomplishment)\b'
        ]

        self.education_patterns = [
            r'\b(education|degree|university|college|school)\b',
            r'\b(master|bachelor|diploma|transcript)\b',
            r'\b(study|studied|studying)\b',
            r'\b(statistics|econometrics|mathematics)\b'
        ]

        self.learning_patterns = [
            r'\b(how|what|where)\s+(to|do|can|should)\s+(learn|study|start|begin|get)\b',
            r'\b(learn|study|practice|improve)\s+(data|python|sql|ai|ml)\b',
            r'\b(advice|guidance|recommendation|tips?)\b',
            r'\b(course|tutorial|training|resource)\b'
        ]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for keyword matching."""
        return text.lower().strip()

    def _calculate_relevance_score(self, text: str, keywords: list, patterns: list = None) -> float:
        """Calculate relevance score for a category."""
        normalized_text = self._normalize_text(text)

        # Keyword matching
        keyword_score = sum(1 for keyword in keywords if keyword in normalized_text)

        # Pattern matching
        pattern_score = 0
        if patterns:
            pattern_score = sum(1 for pattern in patterns if re.search(pattern, normalized_text, re.IGNORECASE))

        return keyword_score + pattern_score

    def route_query(self, user_input: str, chat_history: str = "", session_id: str = "") -> Dict[str, Any]:
        """
        Route the query to the most appropriate agent using dynamic guardrails.

        Args:
            user_input: The user's message
            chat_history: Previous conversation history
            session_id: Unique session identifier

        Returns:
            dict: Contains agent instance, agent type, and confidence score
        """
        # Get session data
        session_info = self.session_data.get(session_id, {'redirect_count': 0, 'last_agent': 'redirect'})

        # Use dynamic guardrails for intelligent analysis
        context_messages = []
        if chat_history:
            # Extract last few messages for context (chat_history is now list of tuples)
            # Convert tuples to strings for analysis
            context_messages = [f"Human: {h[0]}\nAssistant: {h[1]}" for h in chat_history[-2:]]

        analysis = self.guardrails.analyze_message(user_input, context_messages)

        # Handle special cases
        if self._is_greeting(user_input):
            return {
                'agent': self.professional_agent,
                'agent_type': 'professional',
                'confidence': 0.8,
                'scores': {'professional': 1, 'education': 0, 'learning': 0, 'off_topic': 0}
            }

        # If confidence is low or off-topic score is high, use redirect
        if analysis['confidence'] < 0.5 or analysis['primary_category'] == 'off_topic':
            return {
                'agent': self.redirect_agent,
                'agent_type': 'redirect',
                'confidence': analysis['confidence'],
                'scores': analysis['scores'],
                'redirect_count': session_info['redirect_count']
            }

        # Route to appropriate agent
        agents = {
            'professional': self.professional_agent,
            'education': self.education_agent,
            'learning': self.learning_agent
        }

        # Handle edge case where primary category might not be in agents
        if analysis['primary_category'] not in agents:
            return {
                'agent': self.redirect_agent,
                'agent_type': 'redirect',
                'confidence': 0.0,
                'scores': analysis['scores'],
                'redirect_count': session_info['redirect_count']
            }

        return {
            'agent': agents[analysis['primary_category']],
            'agent_type': analysis['primary_category'],
            'confidence': analysis['confidence'],
            'scores': analysis['scores']
        }

    def route_with_classification_agent(self, user_input: str, chat_history: List[Tuple[str, str]], session_id: str) -> Dict[str, Any]:
        """
        Route using the advanced ClassificationAgent with internet access.

        Args:
            user_input: The user's message
            chat_history: List of (human, ai) message tuples
            session_id: Session identifier

        Returns:
            Dict with routing decision and agent instance
        """
        try:
            # Get session info for fallback logic
            session_info = self.session_data.get(session_id, {'last_agent': None})

            # Convert chat history to string format for ClassificationAgent
            chat_history_str = "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in chat_history[-5:]])  # Last 5 exchanges

            # Use ClassificationAgent for routing decision
            routing_decision = self.classification_agent.get_routing_decision(
                user_input=user_input,
                chat_history=chat_history_str,
                last_agent=session_info.get('last_agent')
            )

            target_agent = routing_decision['target_agent']

            # Map to agent instances
            agent_mapping = {
                'professional': self.professional_agent,
                'education': self.education_agent,
                'learning': self.learning_agent,
                'redirect': self.redirect_agent
            }

            # Get the appropriate agent
            if target_agent in agent_mapping:
                agent = agent_mapping[target_agent]
                agent_type = target_agent
            else:
                # Fallback to redirect agent
                agent = self.redirect_agent
                agent_type = 'redirect'

            # Update session info
            session_info['last_agent'] = agent_type

            result = {
                'agent': agent,
                'agent_type': agent_type,
                'confidence': routing_decision['confidence'],
                'reasoning': routing_decision['reasoning'],
                'classification_agent_used': True,
                'fallback_applied': routing_decision.get('fallback_applied', False),
                'fallback_reason': routing_decision.get('fallback_reason')
            }

            print(f"🎯 ClassificationAgent routing: {user_input[:50]}... -> {agent_type} (confidence: {routing_decision['confidence']:.2f})")

            return result

        except Exception as e:
            print(f"❌ ClassificationAgent routing failed: {e}")
            # Fallback to legacy dynamic guardrails routing
            return self.route_query(user_input, chat_history, session_id)

    def process_query(self, user_input: str, chat_history: List[Tuple[str, str]] = None, session_id: str = "", user_language: str = "en") -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        """
        Process a query by routing it to the appropriate agent and getting the response.

        Args:
            user_input: The user's message
            chat_history: Previous conversation history
            session_id: Unique session identifier
            user_language: User's preferred language
        """
        # Get or create session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                'redirect_count': 0,
                'last_agent': 'redirect',
                'language': user_language,
                'conversation_start': True
            }

        session_info = self.session_data[session_id]

        # Use ClassificationAgent for intelligent routing
        routing_result = self.route_with_classification_agent(user_input, chat_history, session_id)

        agent = routing_result['agent']
        agent_type = routing_result['agent_type']

        # Handle redirect agent with enhanced logic
        if agent_type == 'redirect':
            redirect_count = routing_result.get('redirect_count', session_info['redirect_count'])

            # Update redirect count first (so should_end_chat logic works correctly)
            redirect_count += 1
            session_info['redirect_count'] = redirect_count

            # Send Google Chat alert if redirect count is high
            if redirect_count >= 3:
                google_chat_alert.send_redirect_limit_alert(session_id, chat_history, redirect_count)

            result = self.redirect_agent.generate_redirect_response(
                user_input,
                chat_history,
                redirect_count,
                session_id
            )

            result['redirect_count'] = redirect_count

            return result

        # Reset redirect count if we successfully routed to a specialist
        session_info['redirect_count'] = 0
        session_info['last_agent'] = agent_type

        # Handle contact actions (Google Chat alerts)
        if self._is_contact_request(user_input):
            contact_type = self._detect_contact_type(user_input)
            if contact_type:
                google_chat_alert.send_contact_alert(contact_type, session_id, chat_history)

        # Invoke the appropriate agent
        try:
            result = agent.invoke({
                "question": user_input,
                "chat_history": chat_history
            })

            # Add metadata to the result
            result['agent_type'] = agent_type
            result['confidence'] = routing_result['confidence']
            result['language'] = user_language

            return result

        except Exception as e:
            # Fallback to redirect agent if there's an error
            print(f"Error with {agent_type} agent: {e}")
            session_info['redirect_count'] += 1

            return self.redirect_agent.generate_redirect_response(
                user_input,
                chat_history,
                session_info['redirect_count'],
                session_id
            )

    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting."""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                    'salut', 'bonjour', 'hola', 'ciao', 'guten tag', 'konnichiwa']
        message_lower = message.lower().strip()
        return any(greeting in message_lower for greeting in greetings)

    def _is_contact_request(self, message: str) -> bool:
        """Check if message indicates intent to contact directly."""
        contact_keywords = ['contact', 'email', 'meeting', 'appointment', 'book', 'schedule', 'call']
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in contact_keywords)

    def _detect_contact_type(self, message: str) -> str:
        """Detect the type of contact request."""
        message_lower = message.lower()

        if any(word in message_lower for word in ['email', 'mail', 'write']):
            return 'email'
        elif any(word in message_lower for word in ['meeting', 'appointment', 'book', 'schedule', 'call']):
            return 'booking'

        return None

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session."""
        session_info = self.session_data.get(session_id, {})
        return {
            'redirect_count': session_info.get('redirect_count', 0),
            'last_agent': session_info.get('last_agent', 'unknown'),
            'language': session_info.get('language', 'en'),
            'conversation_active': session_id in self.session_data
        }

    def reset_session(self, session_id: str):
        """Reset session data."""
        if session_id in self.session_data:
            del self.session_data[session_id]
