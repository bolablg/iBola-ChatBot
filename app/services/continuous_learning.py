"""
Continuous Learning Pipeline with Automated Model Updates.
"""

import asyncio
import json
import os
import statistics
import sys
import threading
from collections import Counter, defaultdict
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

    CONTINUOUS_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUOUS_LEARNING_AVAILABLE = False
    print("Continuous learning service requires langchain-google-genai")


class FeedbackCollector:
    """
    Collects and analyzes user feedback for continuous improvement.
    """

    def __init__(self):
        self.feedback_data = []
        self.user_satisfaction = defaultdict(list)
        self.response_quality = defaultdict(list)
        self.common_issues = Counter()
        self.improvement_suggestions = []

    def add_feedback(
        self,
        session_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comments: str = "",
        response_quality: Optional[int] = None,
    ):
        """Add user feedback."""

        feedback = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "rating": rating,
            "comments": comments,
            "response_quality": response_quality,
        }

        self.feedback_data.append(feedback)

        if rating:
            self.user_satisfaction[feedback_type].append(rating)

        if response_quality:
            self.response_quality[feedback_type].append(response_quality)

        # Extract common issues from comments
        if comments:
            self._analyze_comments(comments)

    def _analyze_comments(self, comments: str):
        """Analyze user comments for common issues."""
        comments_lower = comments.lower()

        issue_keywords = {
            "slow": ["slow", "waiting", "delay", "takes time"],
            "inaccurate": ["wrong", "incorrect", "not right", "mistake"],
            "unclear": ["unclear", "confusing", "not understand", "confused"],
            "incomplete": ["incomplete", "missing", "not enough", "more info"],
            "irrelevant": ["irrelevant", "off topic", "not related", "unrelated"],
            "technical": ["error", "bug", "problem", "issue", "broken"],
        }

        for issue, keywords in issue_keywords.items():
            if any(keyword in comments_lower for keyword in keywords):
                self.common_issues[issue] += 1

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        total_feedback = len(self.feedback_data)

        if total_feedback == 0:
            return {"message": "No feedback collected yet"}

        # Calculate averages
        satisfaction_scores = {}
        quality_scores = {}

        for feedback_type, ratings in self.user_satisfaction.items():
            if ratings:
                satisfaction_scores[feedback_type] = {
                    "average": statistics.mean(ratings),
                    "count": len(ratings),
                    "distribution": Counter(ratings),
                }

        for feedback_type, qualities in self.response_quality.items():
            if qualities:
                quality_scores[feedback_type] = {
                    "average": statistics.mean(qualities),
                    "count": len(qualities),
                    "distribution": Counter(qualities),
                }

        return {
            "total_feedback": total_feedback,
            "date_range": {
                "start": min(f["timestamp"] for f in self.feedback_data),
                "end": max(f["timestamp"] for f in self.feedback_data),
            },
            "satisfaction_scores": satisfaction_scores,
            "response_quality": quality_scores,
            "common_issues": dict(self.common_issues.most_common(5)),
            "feedback_types": Counter(f["type"] for f in self.feedback_data),
        }

    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get most recent feedback entries."""
        return sorted(self.feedback_data, key=lambda x: x["timestamp"], reverse=True)[
            :limit
        ]


class ModelPerformanceTracker:
    """
    Tracks model performance metrics for continuous improvement.
    """

    def __init__(self):
        self.performance_metrics = defaultdict(list)
        self.agent_performance = defaultdict(list)
        self.error_logs = []
        self.response_times = []
        self.accuracy_scores = []

    def log_performance(
        self,
        agent_type: str,
        metric_name: str,
        value: float,
        metadata: Dict[str, Any] = None,
    ):
        """Log a performance metric."""

        metric = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "metric_name": metric_name,
            "value": value,
            "metadata": metadata or {},
        }

        self.performance_metrics[metric_name].append(metric)
        self.agent_performance[agent_type].append(metric)

    def log_response_time(
        self, agent_type: str, response_time: float, query_complexity: str = "medium"
    ):
        """Log response time for performance tracking."""

        self.response_times.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent_type": agent_type,
                "response_time": response_time,
                "query_complexity": query_complexity,
            }
        )

        # Log to performance metrics
        self.log_performance(
            agent_type,
            "response_time",
            response_time,
            {"query_complexity": query_complexity},
        )

    def log_accuracy(
        self,
        agent_type: str,
        accuracy_score: float,
        expected_answer: str = "",
        actual_answer: str = "",
    ):
        """Log accuracy score."""

        self.accuracy_scores.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent_type": agent_type,
                "accuracy_score": accuracy_score,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
            }
        )

        self.log_performance(
            agent_type,
            "accuracy",
            accuracy_score,
            {"expected": expected_answer, "actual": actual_answer},
        )

    def log_error(
        self,
        agent_type: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None,
    ):
        """Log an error for analysis."""

        error = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
        }

        self.error_logs.append(error)

        # Log as performance metric
        self.log_performance(
            agent_type,
            "error_rate",
            1.0,
            {"error_type": error_type, "message": error_message},
        )

    def get_performance_summary(
        self, time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""

        cutoff_time = datetime.now() - time_window
        cutoff_str = cutoff_time.isoformat()

        # Filter metrics within time window
        recent_metrics = {}
        for metric_name, metrics in self.performance_metrics.items():
            recent = [m for m in metrics if m["timestamp"] >= cutoff_str]
            if recent:
                values = [m["value"] for m in recent]
                recent_metrics[metric_name] = {
                    "count": len(values),
                    "average": statistics.mean(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "recent_values": values[-10:],  # Last 10 values
                }

        # Calculate response time statistics
        recent_response_times = [
            rt for rt in self.response_times if rt["timestamp"] >= cutoff_str
        ]
        response_time_stats = {}
        if recent_response_times:
            times = [rt["response_time"] for rt in recent_response_times]
            response_time_stats = {
                "count": len(times),
                "average": statistics.mean(times),
                "median": statistics.median(times),
                "p95": (
                    sorted(times)[int(len(times) * 0.95)]
                    if len(times) > 1
                    else max(times)
                ),
            }

        # Error analysis
        recent_errors = [e for e in self.error_logs if e["timestamp"] >= cutoff_str]
        error_stats = Counter(e["error_type"] for e in recent_errors)

        return {
            "time_window_days": time_window.days,
            "metrics_summary": recent_metrics,
            "response_time_stats": response_time_stats,
            "error_analysis": dict(error_stats.most_common(5)),
            "total_errors": len(recent_errors),
            "data_points": len(recent_response_times),
        }


class ContinuousLearningPipeline:
    """
    Main pipeline for continuous learning and model improvement.
    """

    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.performance_tracker = ModelPerformanceTracker()
        self.learning_tasks = []
        self.is_running = False
        self.learning_thread = None

        if CONTINUOUS_LEARNING_AVAILABLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", temperature=0.3, google_api_key=GEMINI_API_KEY
            )

            # Improvement analysis chain
            self.improvement_prompt = PromptTemplate.from_template("""
            Analyze the following performance data and feedback to identify areas for improvement.

            Performance Metrics:
            {performance_data}

            User Feedback Summary:
            {feedback_summary}

            Common Issues:
            {common_issues}

            Based on this analysis, suggest specific improvements for the chatbot system:

            1. **Prompt Improvements**: Changes to system prompts or agent instructions
            2. **Model Updates**: When to consider model retraining or fine-tuning
            3. **Feature Additions**: New capabilities or features to implement
            4. **Bug Fixes**: Specific issues to address
            5. **Performance Optimizations**: Speed and efficiency improvements

            Provide actionable recommendations:
            """)

            self.improvement_chain = LLMChain(
                llm=self.llm, prompt=self.improvement_prompt, verbose=False
            )

    def start_continuous_learning(self):
        """Start the continuous learning pipeline."""
        if self.is_running:
            print("Continuous learning already running")
            return

        self.is_running = True
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        print("🚀 Continuous learning pipeline started")

    def stop_continuous_learning(self):
        """Stop the continuous learning pipeline."""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        print("🛑 Continuous learning pipeline stopped")

    def _continuous_learning_loop(self):
        """Main loop for continuous learning."""
        while self.is_running:
            try:
                # Analyze performance every hour
                self._analyze_and_improve()

                # Sleep for 1 hour (3600 seconds)
                for _ in range(3600):
                    if not self.is_running:
                        break
                    asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Error in continuous learning loop: {e}")
                asyncio.sleep(300)  # Wait 5 minutes before retrying

    def _analyze_and_improve(self):
        """Analyze performance data and generate improvement suggestions."""
        try:
            # Get recent performance data
            performance_summary = self.performance_tracker.get_performance_summary(
                timedelta(hours=1)
            )
            feedback_summary = self.feedback_collector.get_feedback_summary()

            # Skip if insufficient data
            if performance_summary.get("data_points", 0) < 5:
                return

            # Generate improvement recommendations
            improvement_result = self.improvement_chain.run(
                performance_data=json.dumps(performance_summary, indent=2),
                feedback_summary=json.dumps(feedback_summary, indent=2),
                common_issues=json.dumps(
                    feedback_summary.get("common_issues", {}), indent=2
                ),
            )

            # Store improvement suggestions
            improvement_task = {
                "timestamp": datetime.now().isoformat(),
                "performance_data": performance_summary,
                "feedback_data": feedback_summary,
                "recommendations": improvement_result.strip(),
                "implemented": False,
            }

            self.learning_tasks.append(improvement_task)

            print("📈 Generated improvement recommendations")
            print(
                f"   Performance data points: {performance_summary.get('data_points', 0)}"
            )
            print(f"   Feedback entries: {feedback_summary.get('total_feedback', 0)}")

            # Log the improvement task
            self._log_improvement_task(improvement_task)

        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")

    def _log_improvement_task(self, task: Dict[str, Any]):
        """Log improvement task for tracking."""
        try:
            log_entry = {
                "timestamp": task["timestamp"],
                "type": "improvement_suggestion",
                "performance_metrics": task["performance_data"],
                "feedback_summary": task["feedback_data"],
                "recommendations": (
                    task["recommendations"][:500] + "..."
                    if len(task["recommendations"]) > 500
                    else task["recommendations"]
                ),
            }

            # In a real implementation, this would be saved to a database or file
            print(
                f"💡 New improvement task generated: {task['recommendations'][:100]}..."
            )

        except Exception as e:
            print(f"❌ Error logging improvement task: {e}")

    def get_recent_improvements(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent improvement suggestions."""
        return sorted(self.learning_tasks, key=lambda x: x["timestamp"], reverse=True)[
            :limit
        ]

    def mark_improvement_implemented(
        self, task_index: int, implementation_notes: str = ""
    ):
        """Mark an improvement task as implemented."""
        if 0 <= task_index < len(self.learning_tasks):
            self.learning_tasks[task_index]["implemented"] = True
            self.learning_tasks[task_index][
                "implementation_notes"
            ] = implementation_notes
            self.learning_tasks[task_index][
                "implemented_at"
            ] = datetime.now().isoformat()
            print(f"✅ Improvement task {task_index} marked as implemented")

    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system health report."""
        performance_summary = self.performance_tracker.get_performance_summary(
            timedelta(hours=24)
        )
        feedback_summary = self.feedback_collector.get_feedback_summary()

        recent_improvements = self.get_recent_improvements(3)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": self._calculate_health_score(
                performance_summary, feedback_summary
            ),
            "performance_summary": performance_summary,
            "feedback_summary": feedback_summary,
            "recent_improvements": recent_improvements,
            "active_learning_tasks": len(
                [t for t in self.learning_tasks if not t.get("implemented", False)]
            ),
            "recommendations": self._generate_health_recommendations(
                performance_summary, feedback_summary
            ),
        }

    def _calculate_health_score(self, performance: Dict, feedback: Dict) -> float:
        """Calculate overall system health score (0-100)."""
        score = 50  # Base score

        # Performance factors
        if "response_time_stats" in performance:
            avg_response_time = performance["response_time_stats"].get("average", 5.0)
            # Faster response times increase score
            if avg_response_time < 2.0:
                score += 20
            elif avg_response_time < 5.0:
                score += 10
            else:
                score -= 10

        # Feedback factors
        if "user_satisfaction" in feedback:
            satisfaction_avg = feedback["user_satisfaction"].get("average", 3.0)
            # Higher satisfaction increases score
            score += (satisfaction_avg - 3.0) * 10

        # Error factors
        error_rate = performance.get("error_analysis", {})
        total_errors = sum(error_rate.values())
        if total_errors > 10:
            score -= 15
        elif total_errors > 5:
            score -= 5

        return max(0, min(100, score))

    def _generate_health_recommendations(
        self, performance: Dict, feedback: Dict
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []

        # Response time recommendations
        if "response_time_stats" in performance:
            avg_time = performance["response_time_stats"].get("average", 5.0)
            if avg_time > 5.0:
                recommendations.append(
                    "Optimize response times - consider caching or model optimization"
                )
            elif avg_time > 10.0:
                recommendations.append(
                    "Critical: Response times are too slow - investigate performance bottlenecks"
                )

        # Error rate recommendations
        if "error_analysis" in performance:
            error_counts = performance["error_analysis"]
            if error_counts:
                top_error = max(error_counts.items(), key=lambda x: x[1])
                if top_error[1] > 5:
                    recommendations.append(
                        f"Address frequent {top_error[0]} errors ({top_error[1]} occurrences)"
                    )

        # Feedback recommendations
        if "common_issues" in feedback:
            issues = feedback["common_issues"]
            if issues:
                top_issue = max(issues.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Address user feedback: {top_issue[0]} ({top_issue[1]} reports)"
                )

        if not recommendations:
            recommendations.append(
                "System health is good - continue monitoring performance"
            )

        return recommendations


# Global continuous learning instance
continuous_learning = (
    ContinuousLearningPipeline() if CONTINUOUS_LEARNING_AVAILABLE else None
)
