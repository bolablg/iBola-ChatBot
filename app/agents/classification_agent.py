"""
Classification Agent - Advanced retriever that classifies user messages and routes them appropriately.
This agent has internet access and uses advanced NLP techniques to understand user intent.
"""

import re
from typing import Any, Dict, Tuple

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

import config

from .retrievers import get_classification_retriever

# Classification prompt - always the same query pattern
CLASSIFICATION_PROMPT = """
You are an advanced classification agent with internet access. Your task is to analyze user messages and classify them into specific categories.

CLASSIFICATION CATEGORIES:
1. **PROFESSIONAL** - Questions about work experience, career, projects, skills, job roles, companies, achievements
2. **EDUCATION** - Questions about degrees, studies, academic background, universities, courses, certifications, academic qualifications
3. **LEARNING** - Questions about how to learn skills, tutorials, courses, advice on career development, skill improvement
4. **OUT_OF_CONTEXT** - Politics, personal opinions, entertainment, sports, weather, food, travel, religion, etc.

EDUCATION KEYWORDS TO RECOGNIZE:
- Degree types: master, bachelor's, bachelor, diploma, phd, doctorate, certificate
- Academic terms: university, college, school, institute, academy, faculty
- Study actions: study, studied, studying, graduate, graduated, major, minor
- Academic fields: statistics, econometrics, mathematics, computer science, engineering
- Questions about education: "As tu un master?", "Do you have a bachelor's?", "What did you study?"

PROFESSIONAL KEYWORDS TO RECOGNIZE:
- Work terms: job, position, role, career, employment, work, working
- Company terms: company, firm, organization, startup, corporation
- Project terms: project, task, assignment, responsibility, achievement
- Skill terms: skill, expertise, technology, tool, experience, background

LEARNING KEYWORDS TO RECOGNIZE:
- Learning actions: learn, study, practice, improve, develop, acquire, gain
- Learning resources: course, tutorial, training, guide, book, video, resource
- Learning questions: how to, what to, where to, best way to

CLASSIFICATION RULES:
- Be very specific about education questions - any mention of degrees, studies, universities, or academic qualifications goes to EDUCATION
- Questions like "As tu un master?", "Do you have a bachelor's?", "What degree do you have?" are EDUCATION questions
- Professional questions are about work experience, job roles, companies worked for, projects completed
- Learning questions are about advice on how to acquire skills or career development
- Use internet search only if needed for context, but rely primarily on keyword analysis
- Consider the conversation history for context
- Default to EDUCATION for academic-related questions, not redirect

RESPONSE FORMAT:
Category: [CATEGORY_NAME]
Confidence: [HIGH/MEDIUM/LOW]
Reasoning: [Brief explanation including specific keywords matched]
WebContext: [Any relevant information from web search]

QUESTION: {question}
CHAT_HISTORY: {chat_history}
"""

# Internet search tool
search = DuckDuckGoSearchRun()


class ClassificationAgent:
    """Advanced classification agent with internet access."""

    def __init__(self):
        # Legacy regex patterns for cross-checking
        self.professional_patterns = [
            r"\b(what|which|where)\s+(do|did|does|is|are|was|were)\s+(you|i|he|bolaji)\s+(work|do)\b",
            r"\b(your|his|bolaji.?s)\s+(job|role|position|career)\b",
            r"\b(experience|background|resume|cv)\b",
            r"\b(skill|technology|tool|expertise)\b",
            r"\b(project|achievement|accomplishment)\b",
        ]

        self.education_patterns = [
            r"\b(education|degree|university|college|school)\b",
            r"\b(master|bachelor|diploma|transcript)\b",
            r"\b(study|studied|studying)\b",
            r"\b(statistics|econometrics|mathematics)\b",
        ]

        self.learning_patterns = [
            r"\b(how|what|where)\s+(to|do|can|should)\s+(learn|study|start|begin|get)\b",
            r"\b(learn|study|practice|improve)\s+(data|python|sql|ai|ml)\b",
            r"\b(advice|guidance|recommendation|tips?)\b",
            r"\b(course|tutorial|training|resource)\b",
        ]
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.7,  # Low temperature for consistent classification
            google_api_key=config.GEMINI_API_KEY,
        )

        # Advanced retriever for classification
        self.retriever = get_classification_retriever()

        self.classification_prompt = PromptTemplate.from_template(CLASSIFICATION_PROMPT)

        # Internet search tool
        self.search_tool = DuckDuckGoSearchRun()

        # Create LLM chain for classification
        self.classification_chain = LLMChain(
            llm=self.llm, prompt=self.classification_prompt, verbose=False
        )

        # For backward compatibility with orchestrator
        self.executor = self.classification_chain

    def _regex_cross_check(self, user_input: str) -> str:
        """
        Cross-check classification with legacy regex patterns.
        Returns the category based on regex matching, or None if no match.
        """
        text_lower = user_input.lower()

        # Check education patterns
        for pattern in self.education_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "education"

        # Check learning patterns
        for pattern in self.learning_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "learning"

        # Check professional patterns
        for pattern in self.professional_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "professional"

        return None  # No regex match

    def _web_search(self, query: str) -> str:
        """Perform web search for additional context."""
        try:
            results = search.run(query)
            return results[:1000]  # Limit results length
        except Exception as e:
            return f"Search failed: {str(e)}"

    def classify_message(
        self, user_input: str, chat_history: str = ""
    ) -> Tuple[str, float, str]:
        """
        Classify the user message and return category, confidence, and reasoning.

        Returns:
            Tuple of (category, confidence_score, reasoning)
        """
        try:
            # Prepare input for the agent
            inputs = {"question": user_input, "chat_history": chat_history}

            # Run classification
            result = self.classification_chain.invoke(inputs)

            # Parse the result - LLMChain returns the output directly
            response_text = (
                str(result) if not isinstance(result, dict) else result.get("text", "")
            )

            # Extract category from response
            category = self._extract_category(response_text)
            confidence = self._extract_confidence(response_text)
            reasoning = self._extract_reasoning(response_text)

            return category, confidence, reasoning

        except Exception as e:
            print(f"Classification error: {e}")
            # Fallback to basic classification
            return self._fallback_classification(user_input)

    def _extract_category(self, response: str) -> str:
        """Extract category from agent response."""
        response_lower = response.lower()

        if "professional" in response_lower:
            return "professional"
        elif "education" in response_lower:
            return "education"
        elif "learning" in response_lower:
            return "learning"
        elif "out_of_context" in response_lower or "out of context" in response_lower:
            return "redirect"
        else:
            return "redirect"  # Default fallback

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from agent response."""
        response_lower = response.lower()

        if "high" in response_lower:
            return 0.9
        elif "medium" in response_lower:
            return 0.7
        elif "low" in response_lower:
            return 0.5
        else:
            return 0.6  # Default medium confidence

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from agent response."""
        # Try to find reasoning section
        reasoning_markers = ["reasoning:", "reason:", "explanation:"]
        for marker in reasoning_markers:
            if marker in response.lower():
                parts = response.lower().split(marker, 1)
                if len(parts) > 1:
                    return parts[1].strip()[:200]  # Limit length

        # Return first line as reasoning
        lines = response.strip().split("\n")
        return lines[0][:200] if lines else "Classification completed"

    def _fallback_classification(self, user_input: str) -> Tuple[str, float, str]:
        """Fallback classification using simple keyword matching."""
        text_lower = user_input.lower()

        # Professional keywords
        prof_keywords = [
            "work",
            "job",
            "career",
            "project",
            "skill",
            "experience",
            "company",
            "role",
        ]
        if any(keyword in text_lower for keyword in prof_keywords):
            return "professional", 0.7, "Contains professional keywords"

        # Education keywords
        edu_keywords = [
            "degree",
            "university",
            "study",
            "academic",
            "school",
            "course",
            "education",
        ]
        if any(keyword in text_lower for keyword in edu_keywords):
            return "education", 0.7, "Contains education keywords"

        # Learning keywords
        learn_keywords = [
            "learn",
            "how to",
            "tutorial",
            "course",
            "advice",
            "teach",
            "study",
        ]
        if any(keyword in text_lower for keyword in learn_keywords):
            return "learning", 0.7, "Contains learning keywords"

        # Default to redirect for anything else
        return "redirect", 0.8, "No matching keywords found - likely off-topic"

    def get_agent_type(self):
        """Return agent type identifier."""
        return "classification"

    def get_routing_decision(
        self, user_input: str, chat_history: str = "", last_agent: str = None
    ) -> Dict[str, Any]:
        """
        Make complete routing decision including fallback logic and regex cross-check.

        Args:
            user_input: The user's message
            chat_history: Previous conversation history
            last_agent: The agent that handled the last message

        Returns:
            Dict with routing decision and metadata
        """
        category, confidence, reasoning = self.classify_message(
            user_input, chat_history
        )

        # If classifier wants to redirect, cross-check with regex patterns
        if category == "redirect":
            regex_category = self._regex_cross_check(user_input)
            if regex_category:
                print(
                    f"🔄 Regex cross-check: {user_input[:50]}... -> {regex_category} (overriding redirect)"
                )
                category = regex_category
                reasoning += f" | Regex override: {regex_category}"

        # Apply fallback logic if confidence is low
        if confidence < 0.6 and last_agent and last_agent != "classification":
            final_agent = last_agent
            fallback_reason = f"Low confidence ({confidence:.2f}), routing to last agent: {last_agent}"
        else:
            final_agent = category if category != "redirect" else "redirect"
            fallback_reason = None

        return {
            "target_agent": final_agent,
            "confidence": confidence,
            "reasoning": reasoning,
            "original_category": category,
            "fallback_applied": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "classification_agent": "advanced_with_internet",
            "regex_cross_check": (
                regex_category if "regex_category" in locals() else None
            ),
        }
