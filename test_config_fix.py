#!/usr/bin/env python3
"""
Test script to verify configuration works without API keys.
"""

import os
import sys

# Set dummy environment variables for testing
os.environ['GEMINI_API_KEY'] = 'test_key_placeholder'
os.environ['GCHAT_WEBHOOK_URL'] = 'https://test.webhook.url'

try:
    # Try to import config
    from config import GEMINI_API_KEY, GCHAT_WEBHOOK_URL, validate_config

    print("✅ Configuration loaded successfully!")
    print(f"  GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Not Set'}")
    print(f"  GCHAT_WEBHOOK_URL: {'Set' if GCHAT_WEBHOOK_URL else 'Not Set'}")

    # Try to import main components
    from app.services.language_detection import language_service
    from app.services.google_chat_alert import google_chat_alert

    print("✅ Services imported successfully!")
    print(f"  Language service: {language_service.__class__.__name__}")
    print(f"  Google Chat service: {google_chat_alert.__class__.__name__}")

    print("\n🎉 All components can be imported without errors!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("The configuration needs to be fixed.")
