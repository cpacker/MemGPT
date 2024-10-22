import base64
import os.path
import sys
import time
from email import message_from_bytes

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# NOTE: THIS file it out of date for >=0.5.0

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_PATH = os.path.expanduser("~/.letta/gmail_token.json")
CREDENTIALS_PATH = os.path.expanduser("~/.letta/google_api_credentials.json")

DELAY = 1

MEMGPT_SERVER_URL = "http://127.0.0.1:8283"
MEMGPT_TOKEN = os.getenv("MEMGPT_SERVER_PASS")
assert MEMGPT_TOKEN, f"Missing env variable MEMGPT_SERVER_PASS"
MEMGPT_AGENT_ID = sys.argv[1] if len(sys.argv) > 1 else None
assert MEMGPT_AGENT_ID, f"Missing agent ID (pass as arg)"


def route_reply_to_letta_api(message):
    # send a POST request to a Letta server

    url = f"{MEMGPT_SERVER_URL}/api/agents/{MEMGPT_AGENT_ID}/messages"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {MEMGPT_TOKEN}",
        "content-type": "application/json",
    }
    data = {
        "stream": False,
        "role": "system",
        "message": f"[EMAIL NOTIFICATION] {message}",
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        print("Got response:", response.text)
    except Exception as e:
        print("Sending message failed:", str(e))


def decode_base64url(data):
    """Decode base64, padding being optional."""
    data += "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data)


def parse_email(message):
    """Parse email content using the email library."""
    msg_bytes = decode_base64url(message["raw"])
    email_message = message_from_bytes(msg_bytes)
    return email_message


def process_email(message) -> dict:
    # print(f"New email from {email_message['from']}: {email_message['subject']}")
    email_message = parse_email(message)
    body_plain_all = ""
    body_html_all = ""
    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_type() == "text/plain":
                body_plain = str(part.get_payload(decode=True).decode("utf-8"))
                # print(body_plain)
                body_plain_all += body_plain
            elif part.get_content_type() == "text/html":
                body_html = str(part.get_payload(decode=True).decode("utf-8"))
                # print(body_html)
                body_html_all += body_html
    else:
        body_plain_all = print(email_message.get_payload(decode=True).decode("utf-8"))

    return {
        "from": email_message["from"],
        "subject": email_message["subject"],
        "body": body_plain_all,
    }


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
    seen_ids = set()  # Set to track seen email IDs

    try:
        # Initially populate the seen_ids with all current unread emails
        print("Grabbing initial state...")
        initial_results = service.users().messages().list(userId="me", q="is:unread", maxResults=500).execute()
        initial_messages = initial_results.get("messages", [])
        seen_ids.update(msg["id"] for msg in initial_messages)

        print("Listening...")
        while True:
            results = service.users().messages().list(userId="me", q="is:unread", maxResults=5).execute()
            messages = results.get("messages", [])
            if messages:
                for message in messages:
                    if message["id"] not in seen_ids:
                        seen_ids.add(message["id"])
                        msg = service.users().messages().get(userId="me", id=message["id"], format="raw").execute()

                        # Optionally mark the message as read here if required
                        email_obj = process_email(msg)
                        msg_str = f"New email from {email_obj['from']}: {email_obj['subject']}, body: {email_obj['body'][:100]}"

                        # Hard check to ignore emails unless
                        # if not (
                        #     "email@address" in email_obj["from"]
                        # ):
                        #     print("ignoring")
                        # else:
                        print(msg_str)
                        route_reply_to_letta_api(msg_str)

            time.sleep(DELAY)  # Wait for N seconds before checking again
    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
