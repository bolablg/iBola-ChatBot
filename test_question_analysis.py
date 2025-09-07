#!/usr/bin/env python3
"""
Test script to analyze how questions are categorized by the dynamic guardrails.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_question_analysis():
    """Test how different questions are analyzed by the dynamic guardrails."""

    print("🧪 Testing Question Analysis")
    print("=" * 60)

    # Import the dynamic guardrails
    from app.services.dynamic_guardrails import dynamic_guardrails

    # Test questions
    test_questions = [
        # French questions
        "As tu un master ?",
        "As-tu un master ?",
        "Est-ce que tu as un master ?",
        "Quelle est ton niveau d'études ?",
        "As tu fait des études ?",

        # English questions
        "Do you have a master's degree?",
        "Do you have a master degree?",
        "What is your educational background?",
        "Where did you study?",
        "What degree do you have?",

        # Professional questions (for comparison)
        "What is your professional experience?",
        "Where have you worked?",
        "What are your skills?",

        # Off-topic questions (for comparison)
        "What is the weather like?",
        "What time is it?",
        "How are you feeling?"
    ]

    print("\n📊 Question Analysis Results:")
    print("-" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i:2d}. Question: '{question}'")

        try:
            analysis = dynamic_guardrails.analyze_message(question)
            scores = analysis['scores']
            confidence = analysis['confidence']
            primary = analysis['primary_category']

            print(f"    Primary Category: {primary}")
            print(".2f")
            print(f"    Scores: Professional={scores['professional']:.1f}, "
                  f"Education={scores['education']:.1f}, "
                  f"Learning={scores['learning']:.1f}, "
                  f"Off-topic={scores['off_topic']:.1f}")

            # Check if it would be routed correctly
            if confidence >= 0.5 and primary in ['professional', 'education', 'learning']:
                print("    ✅ Would be routed to: " + primary + " agent")
            else:
                print("    ❌ Would be redirected (low confidence or off-topic)")

        except Exception as e:
            print(f"    ❌ Error analyzing question: {e}")

    print("\n" + "=" * 60)

    # Show the current education keywords
    print("\n📚 Current Education Keywords:")
    print("Education keywords:", sorted(list(dynamic_guardrails.education_keywords)))
    print("\nProfessional keywords:", sorted(list(dynamic_guardrails.professional_keywords)))

    # Test the keyword matching directly
    print("\n🔍 Direct Keyword Matching Test:")
    test_words = ["master", "masters", "degree", "bachelor", "education", "study"]
    for word in test_words:
        in_education = word in dynamic_guardrails.education_keywords
        in_professional = word in dynamic_guardrails.professional_keywords
        print(f"'{word}': Education={in_education}, Professional={in_professional}")

if __name__ == "__main__":
    test_question_analysis()
