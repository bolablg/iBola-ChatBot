"""
Google Chat integration for sending alerts when users want to contact directly.
"""

import json
import os
import sys
from datetime import datetime

import requests

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import GCHAT_WEBHOOK_URL


class GoogleChatAlert:
    """Service for sending alerts to Google Chat when users want to contact directly."""

    def __init__(self):
        # You'll need to set up a Google Chat webhook URL in your config
        # This can be obtained from Google Chat -> Manage webhooks
        self.webhook_url = GCHAT_WEBHOOK_URL

    def send_contact_alert(
        self, contact_type, session_id, chat_history, user_email=None
    ):
        """
        Send an alert to Google Chat when a user wants to contact directly.

        Args:
            contact_type: 'booking' or 'email'
            session_id: Unique session identifier
            chat_history: List of conversation history
            user_email: User's email if provided
        """
        if not self.webhook_url:
            print("Google Chat webhook URL not configured")
            return False

        # Format chat history for display
        formatted_history = self._format_chat_history(chat_history)

        # Create the message payload
        message = self._create_message_payload(
            contact_type, session_id, formatted_history, user_email
        )

        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                print(f"✅ Google Chat alert sent for {contact_type} contact")
                return True
            else:
                print(f"❌ Failed to send Google Chat alert: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error sending Google Chat alert: {e}")
            return False

    def _format_chat_history(self, chat_history):
        """Format chat history for display in Google Chat."""
        if not chat_history:
            return "No conversation history available"

        formatted = []
        for i, (user_msg, bot_msg) in enumerate(
            chat_history[-5:], 1
        ):  # Last 5 exchanges
            formatted.append(f"*{i}. User:* {user_msg}")
            formatted.append(f"*{i}. Assistant:* {bot_msg}")

        return "\n".join(formatted)

    def _create_message_payload(
        self, contact_type, session_id, chat_history, user_email
    ):
        """Create the message payload for Google Chat."""

        contact_icon = "📅" if contact_type == "booking" else "📧"
        contact_action = (
            "book an appointment" if contact_type == "booking" else "send an email"
        )

        message_text = f"""
{contact_icon} *New Contact Request*

A user wants to {contact_action} with you!

*Session ID:* `{session_id}`
*Contact Type:* {contact_type.title()}
*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Recent Conversation:*
```
{chat_history}
```

*Action Required:* Check your {'calendar' if contact_type == 'booking' else 'email'} for the incoming request.
        """

        if user_email:
            message_text += f"\n*User Email:* {user_email}"

        return {"text": message_text}

    def send_redirect_limit_alert(self, session_id, chat_history, redirect_count):
        """Send alert when user has been redirected 3+ times."""
        if not self.webhook_url:
            return False

        formatted_history = self._format_chat_history(chat_history)

        message = {
            "text": f"""
⚠️ *High Redirect Count Alert*

A user has been redirected {redirect_count} times and may need direct assistance.

*Session ID:* `{session_id}`
*Redirect Count:* {redirect_count}
*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Recent Conversation:*
```
{formatted_history}
```

*Recommendation:* Consider reaching out to this user directly.
            """
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending redirect limit alert: {e}")
            return False


# Global instance for easy access
google_chat_alert = GoogleChatAlert()
