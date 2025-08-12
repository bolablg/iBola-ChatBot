import re

def is_in_scope(message: str) -> bool:
    """Check if the user's message is within the scope of the chatbot."""
    # Use regex for whole-word matching to avoid partial matches like 'ai' in 'available'.
    keywords = ["bolaji", "resume", "skill", "experience", "work", "project", "career", "gozem", "data", "position", "ai", "statistics", "analytics", "machine learning", "python", "sql", "google", "chatbot"]
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', message.lower()) for keyword in keywords)
