import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
# SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
SCOPES = ["https://www.googleapis.com/auth/calendar"]

TOKEN_PATH = os.path.expanduser("~/.memgpt/gcal_token.json")
CREDENTIALS_PATH = os.path.expanduser("~/.memgpt/google_api_credentials.json")


def main():
    """Shows basic usage of the Google Calendar API.
    Prints the start and name of the next 10 events on the user's calendar.
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

    try:
        service = build("calendar", "v3", credentials=creds)

        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
        print("Getting the upcoming 10 events")
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        if not events:
            print("No upcoming events found.")
            return

        # Prints the start and name of the next 10 events
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            print(start, event["summary"])

    except HttpError as error:
        print(f"An error occurred: {error}")

    #### Create an event
    # Refer to the Python quickstart on how to setup the environment:
    # https://developers.google.com/calendar/quickstart/python
    # Change the scope to 'https://www.googleapis.com/auth/calendar' and delete any
    # stored credentials.
    # try:
    #     service = build("calendar", "v3", credentials=creds)

    #     event = {
    #         "summary": "GCAL API TEST EVENT",
    #         # "location": "800 Howard St., San Francisco, CA 94103",
    #         "description": "A chance to hear more about Google's developer products.",
    #         "start": {
    #             "dateTime": "2024-04-23T09:00:00-07:00",
    #             "timeZone": "America/Los_Angeles",
    #         },
    #         "end": {
    #             "dateTime": "2024-04-24T17:00:00-07:00",
    #             "timeZone": "America/Los_Angeles",
    #         },
    #         # "recurrence": ["RRULE:FREQ=DAILY;COUNT=2"],
    #         "attendees": [
    #             {"email": "packercharles@gmail.com"},
    #         ],
    #         # "reminders": {
    #         #     "useDefault": False,
    #         #     "overrides": [
    #         #         {"method": "email", "minutes": 24 * 60},
    #         #         {"method": "popup", "minutes": 10},
    #         #     ],
    #         # },
    #     }

    #     event = service.events().insert(calendarId="primary", body=event).execute()
    #     print("Event created: %s" % (event.get("htmlLink")))

    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
