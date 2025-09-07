#!/usr/bin/env python3
"""
Test script for the multi-agent chatbot system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents.orchestrator import AgentOrchestrator

def test_agent_routing():
    """Test the agent routing functionality."""
    print("🧪 Testing Multi-Agent Routing System")
    print("=" * 50)

    orchestrator = AgentOrchestrator()

    test_queries = [
        # Professional queries
        ("What projects have you worked on?", "professional"),
        ("Tell me about your experience with data engineering", "professional"),
        ("What technologies do you use?", "professional"),

        # Education queries
        ("What is your educational background?", "education"),
        ("Where did you study?", "education"),
        ("What degree do you have?", "education"),

        # Learning queries
        ("How can I learn data science?", "learning"),
        ("What skills should I learn for your job?", "learning"),
        ("How do I start learning Python?", "learning"),

        # Off-topic queries (should route to redirect)
        ("What's the weather like?", "redirect"),
        ("Tell me a joke", "redirect"),
        ("How do I cook pasta?", "redirect"),
    ]

    for query, expected_agent in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {expected_agent}")

        routing_result = orchestrator.route_query(query)
        actual_agent = routing_result['agent_type']
        confidence = routing_result['confidence']

        status = "✅" if actual_agent == expected_agent else "❌"
        print(f"Actual: {actual_agent} (confidence: {confidence:.2f}) {status}")

        if actual_agent != expected_agent:
            print(f"   Scores: {routing_result['scores']}")

def test_agent_responses():
    """Test actual agent responses."""
    print("\n🧪 Testing Agent Responses")
    print("=" * 50)

    orchestrator = AgentOrchestrator()

    test_cases = [
        ("Tell me about your work at Gozem", "professional"),
        ("What university did you attend?", "education"),
        ("How can I learn to be a data engineer?", "learning"),
        ("What's your favorite movie?", "redirect"),
    ]

    for query, expected_agent in test_cases:
        print(f"\nQuery: '{query}'")
        print(f"Expected agent: {expected_agent}")

        try:
            result = orchestrator.process_query(query)
            actual_agent = result.get('agent_type', 'unknown')
            answer = result.get('answer', '')[:100] + "..." if len(result.get('answer', '')) > 100 else result.get('answer', '')

            status = "✅" if actual_agent == expected_agent else "❌"
            print(f"Actual agent: {actual_agent} {status}")
            print(f"Response: {answer}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_agent_routing()
    test_agent_responses()
    print("\n🎉 Testing complete!")
