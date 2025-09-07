"""
Dynamic guardrails system that learns from conversation patterns to make more intelligent decisions.
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set


class DynamicGuardrails:
    """Dynamic guardrails that learn from conversation patterns."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.patterns_file = os.path.join(data_dir, "dynamic_patterns.json")
        self.feedback_file = os.path.join(data_dir, "conversation_feedback.json")

        # Load existing patterns or create defaults
        self.patterns = self._load_patterns()
        self.feedback_history = self._load_feedback()

        # Dynamic keyword sets that evolve
        self.professional_keywords = set(self.patterns.get("professional_keywords", []))
        self.education_keywords = set(self.patterns.get("education_keywords", []))
        self.learning_keywords = set(self.patterns.get("learning_keywords", []))
        self.off_topic_indicators = set(self.patterns.get("off_topic_indicators", []))

        # Conversation context tracking
        self.conversation_contexts = defaultdict(list)

    def _load_patterns(self):
        """Load dynamic patterns from file."""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading patterns: {e}")

        # Default patterns
        return {
            "professional_keywords": [
                "experience",
                "work",
                "job",
                "career",
                "project",
                "company",
                "role",
                "position",
                "skill",
                "technology",
                "tool",
                "expertise",
                "achievement",
                "gozem",
                "rintio",
                "data",
                "analytics",
                "ai",
                "machine learning",
                "cloud",
                "bigquery",
                "python",
                "sql",
                "airflow",
                "looker",
                "dataform",
                "leadership",
                "team",
                "automation",
                "optimization",
                "business",
            ],
            "education_keywords": [
                "education",
                "degree",
                "university",
                "college",
                "master",
                "bachelor",
                "diploma",
                "study",
                "academic",
                "school",
                "statistics",
                "econometrics",
                "icmpa",
                "unesco",
                "transcript",
                "grade",
                "gpa",
                "thesis",
                "dissertation",
            ],
            "learning_keywords": [
                "learn",
                "study",
                "how to",
                "tutorial",
                "course",
                "training",
                "advice",
                "guide",
                "beginner",
                "start",
                "career path",
                "skill development",
                "resources",
                "practice",
                "improve",
                "tips",
                "recommend",
            ],
            "off_topic_indicators": [
                "weather",
                "sports",
                "politics",
                "election",
                "president",
                "government",
                "political",
                "vote",
                "voting",
                "candidate",
                "party",
                "democracy",
                "religion",
                "religious",
                "faith",
                "belief",
                "church",
                "temple",
                "entertainment",
                "celebrity",
                "gossip",
                "food",
                "restaurant",
                "recipe",
                "travel",
                "vacation",
                "hotel",
                "shopping",
                "store",
                "purchase",
                "music",
                "song",
                "album",
                "concert",
                "band",
                "artist",
                "movies",
                "film",
                "cinema",
                "actor",
                "actress",
                "director",
                "games",
                "gaming",
                "video game",
                "console",
                "playstation",
                "xbox",
                "football",
                "soccer",
                "basketball",
                "baseball",
                "tennis",
                "golf",
                "cricket",
                "rugby",
                "hockey",
                "swimming",
                "athletics",
                "olympics",
            ],
        }

    def _load_feedback(self):
        """Load conversation feedback history."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback: {e}")
        return []

    def _save_patterns(self):
        """Save updated patterns to file."""
        try:
            patterns_data = {
                "professional_keywords": list(self.professional_keywords),
                "education_keywords": list(self.education_keywords),
                "learning_keywords": list(self.learning_keywords),
                "off_topic_indicators": list(self.off_topic_indicators),
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.patterns_file, "w") as f:
                json.dump(patterns_data, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")

    def _save_feedback(self):
        """Save feedback history."""
        try:
            # Keep only last 1000 feedback entries
            recent_feedback = self.feedback_history[-1000:]
            with open(self.feedback_file, "w") as f:
                json.dump(recent_feedback, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback: {e}")

    def analyze_message(self, message: str, context: List[str] = None) -> Dict:
        """
        Analyze a message and determine its relevance and category.

        Args:
            message: The user message to analyze
            context: Previous messages in the conversation

        Returns:
            dict: Analysis results with scores and recommendations
        """
        normalized_message = self._normalize_text(message)

        # Calculate scores for each category
        scores = {
            "professional": self._calculate_category_score(
                normalized_message, self.professional_keywords
            ),
            "education": self._calculate_category_score(
                normalized_message, self.education_keywords
            ),
            "learning": self._calculate_category_score(
                normalized_message, self.learning_keywords
            ),
            "off_topic": self._calculate_category_score(
                normalized_message, self.off_topic_indicators
            ),
        }

        # Consider conversation context
        if context:
            context_scores = self._analyze_context(context, normalized_message)
            scores = self._combine_scores(scores, context_scores)

        # Determine primary category with special handling for off-topic
        off_topic_score = scores.get("off_topic", 0)

        # If off-topic score is significant, prioritize redirect
        if off_topic_score > 2 or (
            off_topic_score > 1 and off_topic_score >= max(scores.values()) * 0.8
        ):
            primary_category = "redirect"
            # Add redirect score to the scores dict for confidence calculation
            scores["redirect"] = off_topic_score
        else:
            primary_category = max(scores, key=scores.get)

        confidence = (
            scores[primary_category] / sum(scores.values())
            if sum(scores.values()) > 0
            else 0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(scores, confidence, context)

        return {
            "primary_category": primary_category,
            "confidence": confidence,
            "scores": scores,
            "recommendations": recommendations,
            "needs_human_review": self._needs_human_review(scores, confidence),
        }

    def _calculate_category_score(self, message: str, keywords: Set[str]) -> float:
        """Calculate relevance score for a category."""
        score = 0
        words = set(message.split())

        # Exact keyword matches (higher weight)
        exact_matches = len(keywords.intersection(words))
        score += exact_matches * 3  # Increased from 2 to 3

        # Partial matches and patterns
        for keyword in keywords:
            if keyword in message:
                score += 1.5  # Increased from 1 to 1.5

        # Boost for question patterns
        if any(
            word in message for word in ["what", "how", "where", "when", "why", "who"]
        ):
            score += 0.5

        # Special handling for political topics
        political_indicators = [
            "election",
            "president",
            "vote",
            "political",
            "government",
            "party",
            "candidate",
        ]
        if any(indicator in message for indicator in political_indicators):
            score += 2  # Strong boost for political content

        return score

    def _analyze_context(self, context: List[str], current_message: str) -> Dict:
        """Analyze conversation context to improve categorization."""
        context_scores = {
            "professional": 0,
            "education": 0,
            "learning": 0,
            "off_topic": 0,
        }

        # Look for topic continuity in recent messages
        recent_context = context[-3:] if len(context) > 3 else context
        context_text = " ".join(recent_context).lower()

        # Boost scores based on conversation flow
        for category, keywords in [
            ("professional", self.professional_keywords),
            ("education", self.education_keywords),
            ("learning", self.learning_keywords),
            ("off_topic", self.off_topic_indicators),
        ]:
            context_matches = sum(1 for keyword in keywords if keyword in context_text)
            if context_matches > 0:
                context_scores[category] += context_matches * 0.3

        return context_scores

    def _combine_scores(self, base_scores: Dict, context_scores: Dict) -> Dict:
        """Combine base and context scores intelligently."""
        combined = {}
        for category in base_scores:
            # Base score has higher weight, context provides boost
            combined[category] = base_scores[category] + (
                context_scores[category] * 0.4
            )
        return combined

    def _generate_recommendations(
        self, scores: Dict, confidence: float, context: List[str]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if confidence < 0.6:
            recommendations.append("Consider asking for clarification")
            recommendations.append("This query might benefit from rephrasing")

        # Check for mixed signals
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > sorted_scores[0][1] * 0.7:
            recommendations.append("Query might be relevant to multiple categories")

        # Context-based recommendations
        if context and len(context) > 5:
            recommendations.append("Long conversation detected - consider summarizing")

        return recommendations

    def _needs_human_review(self, scores: Dict, confidence: float) -> bool:
        """Determine if message needs human review."""
        # Low confidence queries need review
        if confidence < 0.5:
            return True

        # Very high off-topic scores need review
        if scores.get("off_topic", 0) > 3:
            return True

        # Mixed signals need review
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1] > sorted_scores[0] * 0.8:
            return True

        return False

    def learn_from_feedback(
        self, message: str, correct_category: str, context: List[str] = None
    ):
        """
        Learn from user feedback to improve future categorizations.

        Args:
            message: The original message
            correct_category: The correct category as determined by user feedback
            context: Conversation context
        """
        normalized_message = self._normalize_text(message)
        words = set(normalized_message.split())

        # Add relevant words to the correct category
        if correct_category == "professional":
            self.professional_keywords.update(words)
        elif correct_category == "education":
            self.education_keywords.update(words)
        elif correct_category == "learning":
            self.learning_keywords.update(words)
        elif correct_category == "off_topic":
            self.off_topic_indicators.update(words)

        # Remove from other categories to avoid confusion
        other_categories = ["professional", "education", "learning", "off_topic"]
        other_categories.remove(correct_category)

        for category in other_categories:
            if category == "professional":
                self.professional_keywords -= words
            elif category == "education":
                self.education_keywords -= words
            elif category == "learning":
                self.learning_keywords -= words
            elif category == "off_topic":
                self.off_topic_indicators -= words

        # Record feedback for analysis
        feedback_entry = {
            "message": message,
            "correct_category": correct_category,
            "timestamp": datetime.now().isoformat(),
            "context_length": len(context) if context else 0,
        }
        self.feedback_history.append(feedback_entry)

        # Save updated patterns
        self._save_patterns()
        self._save_feedback()

    def get_statistics(self) -> Dict:
        """Get statistics about the dynamic guardrails performance."""
        total_feedback = len(self.feedback_history)
        category_distribution = Counter(
            [f["correct_category"] for f in self.feedback_history]
        )

        return {
            "total_feedback_entries": total_feedback,
            "category_distribution": dict(category_distribution),
            "keyword_counts": {
                "professional": len(self.professional_keywords),
                "education": len(self.education_keywords),
                "learning": len(self.learning_keywords),
                "off_topic": len(self.off_topic_indicators),
            },
            "last_updated": datetime.now().isoformat(),
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


# Global instance
dynamic_guardrails = DynamicGuardrails()
