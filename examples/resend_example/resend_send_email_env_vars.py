import json
import os

import requests


def send_email(self, description: str):
    """
    Sends an email to a predefined user. The email contains a message, which is defined by the description parameter.

    Args:
        description (str): Email contents. All unicode (including emojis) are supported.

    Returns:
        None

    Example:
        >>> send_email("hello")
        # Output: None. This will send an email to the you are talking to with the message "hello".
    """
    RESEND_API_KEY = os.getenv("RESEND_API_KEY")
    RESEND_TARGET_EMAIL_ADDRESS = os.getenv("RESEND_TARGET_EMAIL_ADDRESS")
    if RESEND_API_KEY is None:
        raise Exception("User did not set the environment variable RESEND_API_KEY")
    if RESEND_TARGET_EMAIL_ADDRESS is None:
        raise Exception("User did not set the environment variable RESEND_TARGET_EMAIL_ADDRESS")

    url = "https://api.resend.com/emails"
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}
    data = {
        "from": "onboarding@resend.dev",
        "to": RESEND_TARGET_EMAIL_ADDRESS,
        "subject": "Letta message:",
        "html": f"<strong>{description}</strong>",
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response.text)
    except requests.HTTPError as e:
        raise Exception(f"send_email failed with an HTTP error: {str(e)}")
    except Exception as e:
        raise Exception(f"send_email failed with an error: {str(e)}")
