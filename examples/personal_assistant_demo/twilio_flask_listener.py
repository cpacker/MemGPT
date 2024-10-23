import os
import sys

import requests
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


app = Flask(__name__)
CORS(app)

# NOTE: this is out of date for >=0.5.0

MEMGPT_SERVER_URL = "http://127.0.0.1:8283"
MEMGPT_TOKEN = os.getenv("MEMGPT_SERVER_PASS")
assert MEMGPT_TOKEN, f"Missing env variable MEMGPT_SERVER_PASS"
MEMGPT_AGENT_ID = sys.argv[1] if len(sys.argv) > 1 else None
assert MEMGPT_AGENT_ID, f"Missing agent ID (pass as arg)"


@app.route("/test", methods=["POST"])
def test():
    print(request.headers)
    return "Headers received. Check your console."


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
        "message": f"[SMS MESSAGE NOTIFICATION - you MUST use send_text_message NOT send_message if you want to reply to the text thread] {message}",
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        print("Got response:", response.text)
    except Exception as e:
        print("Sending message failed:", str(e))


@app.route("/sms", methods=["POST"])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Fetch the message
    message_body = request.form["Body"]
    from_number = request.form["From"]

    # print(f"New message from {from_number}: {message_body}")
    msg_str = f"New message from {from_number}: {message_body}"
    print(msg_str)

    route_reply_to_letta_api(msg_str)
    return str("status = OK")

    # Start our response
    # resp = MessagingResponse()

    # Add a message
    # resp.message("Hello, thanks for messaging!")

    # return str(resp)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8284, debug=True)
