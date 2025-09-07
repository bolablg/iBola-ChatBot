#!/usr/bin/env python3
"""
Test script to directly test the education agent with master's degree questions.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables
os.environ['GEMINI_API_KEY'] = 'test_key_placeholder'

def test_education_agent():
    """Test the education agent directly with master's degree questions."""

    print("🧪 Testing Education Agent Directly")
    print("=" * 60)

    try:
        from app.agents.education_agent import EducationAgent

        # Initialize the agent
        print("Initializing Education Agent...")
        agent = EducationAgent()
        print("✅ Education Agent initialized successfully")

        # Test questions
        test_questions = [
            "Do you have a master's degree?",
            "Do you have a master degree?",
            "As tu un master ?",
            "What is your educational background?",
            "What degree do you have?"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: '{question}'")
            print("-" * 40)

            try:
                # Test the retriever first
                docs = agent.retriever.get_relevant_documents(question)
                print(f"📄 Found {len(docs)} relevant documents")

                if docs:
                    print("📝 Document content preview:")
                    for j, doc in enumerate(docs[:2]):  # Show first 2 documents
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        print(f"   Doc {j+1}: {content}")

                # Test the full agent response
                print("\n🤖 Agent Response:")
                result = agent.get_response(question, [])
                answer = result.get('answer', 'No answer generated')
                print(f"   Answer: {answer}")

                # Check if master's degree info is mentioned
                if 'master' in answer.lower() or 'degree' in answer.lower():
                    print("   ✅ Master's degree information found in response")
                else:
                    print("   ⚠️  Master's degree information NOT found in response")

            except Exception as e:
                print(f"   ❌ Error processing question: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("🎯 Analysis Complete")

    except Exception as e:
        print(f"❌ Error initializing Education Agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_education_agent()
