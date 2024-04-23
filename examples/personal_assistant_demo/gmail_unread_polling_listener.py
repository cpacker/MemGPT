import os.path
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_PATH = os.path.expanduser("~/.memgpt/gmail_token.json")
CREDENTIALS_PATH = os.path.expanduser("~/.memgpt/google_api_credentials.json")

DELAY = 1


def main():
    """Monitors for new emails and prints their titles."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    last_checked = None
    seen_ids = set()  # Set to track seen email IDs

    try:
        # Initially populate the seen_ids with all current unread emails
        initial_results = service.users().messages().list(userId="me", q="is:unread", maxResults=500).execute()
        initial_messages = initial_results.get("messages", [])
        seen_ids.update(msg["id"] for msg in initial_messages)

        while True:
            results = service.users().messages().list(userId="me", q="is:unread", maxResults=5).execute()
            messages = results.get("messages", [])
            if messages:
                for message in messages:
                    if message["id"] not in seen_ids:
                        seen_ids.add(message["id"])
                        msg = service.users().messages().get(userId="me", id=message["id"], format="metadata").execute()
                        headers = msg.get("payload", {}).get("headers", [])
                        subject = next(header["value"] for header in headers if header["name"] == "Subject")
                        print(f"New email: {subject}")
                        # Optionally mark the message as read here if required
            time.sleep(DELAY)  # Wait for 30 seconds before checking again
    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
