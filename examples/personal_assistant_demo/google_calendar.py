# Enabling API control on Google Calendar requires a few steps:
# https://developers.google.com/calendar/api/quickstart/python
# including:
#   pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

import os
import os.path
import traceback
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
# SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_PATH = os.path.expanduser("~/.letta/gcal_token.json")
CREDENTIALS_PATH = os.path.expanduser("~/.letta/google_api_credentials.json")


def schedule_event(
    self,
    title: str,
    start: str,
    end: str,
    # attendees: Optional[List[str]] = None,
    # attendees: Optional[list[str]] = None,
    description: Optional[str] = None,
    # timezone: Optional[str] = "America/Los_Angeles",
) -> str:
    """
    Schedule an event on the user's Google Calendar. Start and end time must be in ISO 8601 format, e.g. February 1st 2024 at noon PT would be "2024-02-01T12:00:00-07:00".

    Args:
        title (str): Event name
        start (str): Start time in ISO 8601 format (date, time, and timezone offset)
        end (str): End time in ISO 8601 format (date, time, and timezone offset)
        description (Optional[str]): Expanded description of the event

    Returns:
        str: The status of the event scheduling request.
    """

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    #### Create an event
    # Refer to the Python quickstart on how to setup the environment:
    # https://developers.google.com/calendar/quickstart/python
    # Change the scope to 'https://www.googleapis.com/auth/calendar' and delete any
    # stored credentials.
    try:
        service = build("calendar", "v3", credentials=creds)

        event = {
            "summary": title,
            # "location": "800 Howard St., San Francisco, CA 94103",
            "start": {
                "dateTime": start,
                "timeZone": "America/Los_Angeles",
            },
            "end": {
                "dateTime": end,
                "timeZone": "America/Los_Angeles",
            },
        }

        # if attendees is not None:
        # event["attendees"] = attendees

        if description is not None:
            event["description"] = description

        event = service.events().insert(calendarId="primary", body=event).execute()
        return "Event created: %s" % (event.get("htmlLink"))

    except HttpError as error:
        traceback.print_exc()

        return f"An error occurred while trying to create an event: {str(error)}"
