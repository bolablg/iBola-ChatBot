"""
Google Sheets logging service for redirect analysis.
Logs all redirected messages with detailed information for classifier improvement.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GCP_SA_CREDENTIALS_PATH, GCP_PROJECT_ID

try:
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("Google Sheets API not available. Install with: pip install google-api-python-client google-auth")

class GoogleSheetsLogger:
    """
    Service for logging redirect events to Google Sheets for classifier analysis.
    """

    def __init__(self, spreadsheet_id: Optional[str] = None):
        self.spreadsheet_id = spreadsheet_id or os.getenv("REDIRECT_LOG_SHEET_ID")
        self.service = None
        self.sheet_name = "Redirects"

        if GOOGLE_SHEETS_AVAILABLE and GCP_SA_CREDENTIALS_PATH:
            self._initialize_service()

    def _initialize_service(self):
        """Initialize Google Sheets API service."""
        try:
            if not os.path.exists(GCP_SA_CREDENTIALS_PATH):
                print("❌ Service account credentials not found for Google Sheets")
                return

            # Set credentials path
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_SA_CREDENTIALS_PATH

            # Create credentials
            creds = Credentials.from_service_account_file(
                GCP_SA_CREDENTIALS_PATH,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )

            # Build the service
            self.service = build('sheets', 'v4', credentials=creds)

            # Ensure the sheet exists and has headers
            self._ensure_sheet_exists()

            print("✅ Google Sheets logging service initialized")

        except Exception as e:
            print(f"❌ Failed to initialize Google Sheets service: {e}")
            self.service = None

    def _ensure_sheet_exists(self):
        """Ensure the redirect logging sheet exists with proper headers."""
        if not self.service or not self.spreadsheet_id:
            return

        try:
            # Check if sheet exists
            sheet_metadata = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()

            sheet_exists = any(
                sheet['properties']['title'] == self.sheet_name
                for sheet in sheet_metadata.get('sheets', [])
            )

            if not sheet_exists:
                # Create the sheet
                requests = [{
                    'addSheet': {
                        'properties': {
                            'title': self.sheet_name
                        }
                    }
                }]

                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': requests}
                ).execute()

                print(f"📊 Created new sheet: {self.sheet_name}")

            # Add headers if sheet is empty
            range_name = f"{self.sheet_name}!A1:Z1"
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            if not result.get('values'):
                # Add headers
                headers = [
                    [
                        "Timestamp",
                        "Session ID",
                        "IP Address",
                        "User Agent",
                        "Browser Language",
                        "User Language",
                        "User Input",
                        "Redirect Count",
                        "Agent Type",
                        "Confidence Score",
                        "Redirect Reason",
                        "Chat History Summary",
                        "Response Time (ms)",
                        "Source Documents Count",
                        "Cache Hit",
                        "Geolocation",
                        "Device Type",
                        "Referrer"
                    ]
                ]

                self.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='RAW',
                    body={'values': headers}
                ).execute()

                print("📋 Added headers to redirect logging sheet")

        except Exception as e:
            print(f"❌ Error ensuring sheet exists: {e}")

    def log_redirect_event(self, redirect_data: Dict[str, Any]) -> bool:
        """
        Log a redirect event to Google Sheets.

        Args:
            redirect_data: Dictionary containing redirect information

        Returns:
            bool: True if logging was successful, False otherwise
        """
        if not self.service or not self.spreadsheet_id:
            print("⚠️  Google Sheets service not available for redirect logging")
            return False

        try:
            # Prepare the row data
            row_data = [
                redirect_data.get('timestamp', datetime.now().isoformat()),
                redirect_data.get('session_id', ''),
                redirect_data.get('ip_address', ''),
                redirect_data.get('user_agent', ''),
                redirect_data.get('browser_language', ''),
                redirect_data.get('user_language', ''),
                redirect_data.get('user_input', ''),
                redirect_data.get('redirect_count', 0),
                redirect_data.get('agent_type', ''),
                redirect_data.get('confidence', 0.0),
                redirect_data.get('redirect_reason', ''),
                redirect_data.get('chat_history_summary', ''),
                redirect_data.get('response_time', 0),
                redirect_data.get('source_documents_count', 0),
                redirect_data.get('cache_hit', False),
                redirect_data.get('geolocation', ''),
                redirect_data.get('device_type', ''),
                redirect_data.get('referrer', '')
            ]

            # Convert all values to strings to avoid type issues
            row_data = [str(value) for value in row_data]

            # Append the row to the sheet
            range_name = f"{self.sheet_name}!A:A"  # Append to column A (will auto-expand)

            body = {
                'values': [row_data]
            }

            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            print(f"📊 Logged redirect event for session {redirect_data.get('session_id', 'unknown')}")
            return True

        except Exception as e:
            print(f"❌ Error logging redirect to Google Sheets: {e}")
            return False

    def get_redirect_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get statistics about recent redirects."""
        if not self.service or not self.spreadsheet_id:
            return {"error": "Google Sheets service not available"}

        try:
            # Get all data from the sheet
            range_name = f"{self.sheet_name}!A:Z"
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            rows = result.get('values', [])
            if len(rows) <= 1:  # Only headers or empty
                return {"total_redirects": 0, "message": "No redirect data available"}

            # Remove header row
            data_rows = rows[1:]

            # Filter by recent days
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

            recent_redirects = []
            for row in data_rows:
                if len(row) >= 1:
                    try:
                        row_timestamp = datetime.fromisoformat(row[0])
                        if row_timestamp >= cutoff_date:
                            recent_redirects.append(row)
                    except (ValueError, IndexError):
                        continue

            # Calculate statistics
            total_redirects = len(recent_redirects)

            if total_redirects == 0:
                return {"total_redirects": 0, "message": f"No redirects in the last {days} days"}

            # Count by agent type (column 8)
            agent_types = {}
            redirect_reasons = {}
            languages = {}

            for row in recent_redirects:
                if len(row) >= 11:  # Make sure we have enough columns
                    agent_type = row[8] if len(row) > 8 else "unknown"
                    redirect_reason = row[10] if len(row) > 10 else "unknown"
                    language = row[5] if len(row) > 5 else "unknown"

                    agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                    redirect_reasons[redirect_reason] = redirect_reasons.get(redirect_reason, 0) + 1
                    languages[language] = languages.get(language, 0) + 1

            return {
                "total_redirects": total_redirects,
                "period_days": days,
                "agent_type_distribution": agent_types,
                "redirect_reason_distribution": redirect_reasons,
                "language_distribution": languages,
                "average_redirects_per_day": total_redirects / days
            }

        except Exception as e:
            print(f"❌ Error getting redirect stats: {e}")
            return {"error": f"Failed to get stats: {str(e)}"}

    def create_analysis_report(self) -> Dict[str, Any]:
        """Create a comprehensive analysis report of redirect patterns."""
        stats = self.get_redirect_stats(days=30)

        if stats.get("total_redirects", 0) == 0:
            return {"message": "Insufficient data for analysis"}

        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_period_days": 30,
            **stats,
            "insights": []
        }

        # Generate insights
        agent_types = stats.get("agent_type_distribution", {})
        if agent_types:
            most_common_agent = max(agent_types.items(), key=lambda x: x[1])
            report["insights"].append(
                f"Most redirects come from {most_common_agent[0]} agent ({most_common_agent[1]} redirects)"
            )

        redirect_reasons = stats.get("redirect_reason_distribution", {})
        if redirect_reasons:
            most_common_reason = max(redirect_reasons.items(), key=lambda x: x[1])
            report["insights"].append(
                f"Most common redirect reason: {most_common_reason[0]} ({most_common_reason[1]} occurrences)"
            )

        # Language analysis
        languages = stats.get("language_distribution", {})
        if languages:
            language_insights = []
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                if count > stats["total_redirects"] * 0.1:  # More than 10%
                    language_insights.append(f"{lang}: {count}")
            if language_insights:
                report["insights"].append(f"Languages with significant redirects: {', '.join(language_insights)}")

        return report

# Global instance
google_sheets_logger = GoogleSheetsLogger() if GOOGLE_SHEETS_AVAILABLE else None
