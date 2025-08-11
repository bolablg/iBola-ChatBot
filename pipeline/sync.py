import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import json
import httpx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import config
from pipeline.update_vectorstore import update_vectorstore

# If modifying these SCOPES, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SYNC_STATE_FILE = os.path.join(config.DATA_PATH, ".sync_state.json")

def send_webhook_alert(updated_files):
    """Sends a message to a Google Chat webhook."""
    webhook_url = config.GCHAT_WEBHOOK_URL
    if webhook_url:
        if updated_files:
            message = {
                "cards": [
                    {
                        "header": {
                            "title": "Vector Store Update",
                            "subtitle": "The following files have been updated:"
                        },
                        "sections": [
                            {
                                "widgets": [
                                    {
                                        "textParagraph": {
                                            "text": "\n".join(updated_files)
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        else:
            message = {
                "cards": [
                    {
                        "header": {
                            "title": "Vector Store Update",
                            "subtitle": "No changes detected in the Google Drive folder."
                        }
                    }
                ]
            }
        with httpx.Client() as client:
            response = client.post(webhook_url, json=message)
            response.raise_for_status()

def get_sync_state():
    if os.path.exists(SYNC_STATE_FILE):
        with open(SYNC_STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sync_state(state):
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f)

def sync_folder_recursive(service, folder_id, local_path, sync_state):
    """Syncs a folder and its contents recursively."""
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    results = (
        service.files()
        .list(q=f"'{folder_id}' in parents", fields="nextPageToken, files(id, name, mimeType, modifiedTime)")
        .execute()
    )
    items = results.get("files", [])

    updated_files = []
    for item in items:
        item_path = os.path.join(local_path, item["name"])
        if item["mimeType"] == "application/vnd.google-apps.folder":
            updated_files.extend(sync_folder_recursive(service, item["id"], item_path, sync_state))
        else:
            if item["id"] not in sync_state or item["modifiedTime"] > sync_state.get(item["id"]):
                if item["mimeType"].startswith("application/vnd.google-apps"):
                    if item["mimeType"] == "application/vnd.google-apps.document":
                        request = service.files().export_media(fileId=item["id"], mimeType="application/pdf")
                        item_path += ".pdf"
                    elif item["mimeType"] == "application/vnd.google-apps.spreadsheet":
                        request = service.files().export_media(fileId=item["id"], mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        item_path += ".xlsx"
                    elif item["mimeType"] == "application/vnd.google-apps.presentation":
                        request = service.files().export_media(fileId=item["id"], mimeType="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                        item_path += ".pptx"
                    else:
                        continue
                else:
                    request = service.files().get_media(fileId=item["id"])
                
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Download {item['name']} {int(status.progress() * 100)}%.")
                with open(item_path, "wb") as f:
                    f.write(fh.getbuffer())
                
                sync_state[item["id"]] = item["modifiedTime"]
                updated_files.append(item['name'])
    return updated_files

def sync_google_drive():
    """Syncs a Google Drive folder with a local folder."""
    creds = None
    if os.path.exists("_conf/token.json"):
        creds = Credentials.from_authorized_user_file("_conf/token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                config.GOOGLE_OAUTH_CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=8080)
        with open("_conf/token.json", "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)

    sync_state = get_sync_state()
    updated_files = sync_folder_recursive(service, config.GDRIVE_FOLDER_ID, config.DATA_PATH, sync_state)

    if updated_files:
        print("Changes detected, updating vector store...")
        update_vectorstore()
        save_sync_state(sync_state)
        send_webhook_alert(updated_files)
    else:
        print("No changes detected.")
        send_webhook_alert([])

if __name__ == "__main__":
    sync_google_drive()