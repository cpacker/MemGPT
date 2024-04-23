# Download the helper library from https://www.twilio.com/docs/python/install
import os
import traceback

from twilio.rest import Client


def send_text_message(self, message: str) -> str:
    """
    Sends an SMS message to the user's phone / cellular device.

    Args:
        message (str): The contents of the message to send.

    Returns:
        str: The status of the text message.
    """
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    client = Client(account_sid, auth_token)

    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("TWILIO_TO_NUMBER")
    assert from_number and to_number
    # assert from_number.startswith("+1") and len(from_number) == 12, from_number
    # assert to_number.startswith("+1") and len(to_number) == 12, to_number

    try:
        message = client.messages.create(
            body=str(message),
            from_=from_number,
            to=to_number,
        )
        return "Message was successfully sent."

    except Exception as e:
        traceback.print_exc()

        return f"Message failed to send with error: {str(e)}"
