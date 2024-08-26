import os
import uuid
from typing import Optional

import requests

from memgpt.constants import (
    MESSAGE_CHATGPT_FUNCTION_MODEL,
    MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE,
)
from memgpt.llm_api.llm_api_tools import create
from memgpt.schemas.message import Message
from memgpt.utils import json_dumps, json_loads


def message_chatgpt(self, message: str):
    """
    Send a message to a more basic AI, ChatGPT. A useful resource for asking questions. ChatGPT does not retain memory of previous interactions.

    Args:
        message (str): Message to send ChatGPT. Phrase your message as a full English sentence.

    Returns:
        str: Reply message from ChatGPT
    """
    dummy_user_id = uuid.uuid4()
    dummy_agent_id = uuid.uuid4()
    message_sequence = [
        Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="system", text=MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE),
        Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="user", text=str(message)),
    ]
    # TODO: this will error without an LLMConfig
    response = create(
        model=MESSAGE_CHATGPT_FUNCTION_MODEL,
        messages=message_sequence,
    )

    reply = response.choices[0].message.content
    return reply


def read_from_text_file(self, filename: str, line_start: int, num_lines: Optional[int] = 1):
    """
    Read lines from a text file.

    Args:
        filename (str): The name of the file to read.
        line_start (int): Line to start reading from.
        num_lines (Optional[int]): How many lines to read (defaults to 1).

    Returns:
        str: Text read from the file
    """
    max_chars = 500
    trunc_message = True
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    if line_start < 1 or num_lines < 1:
        raise ValueError("Both line_start and num_lines must be positive integers.")

    lines = []
    chars_read = 0
    with open(filename, "r", encoding="utf-8") as file:
        for current_line_number, line in enumerate(file, start=1):
            if line_start <= current_line_number < line_start + num_lines:
                chars_to_add = len(line)
                if max_chars is not None and chars_read + chars_to_add > max_chars:
                    # If adding this line exceeds MAX_CHARS, truncate the line if needed and stop reading further.
                    excess_chars = (chars_read + chars_to_add) - max_chars
                    lines.append(line[:-excess_chars].rstrip("\n"))
                    if trunc_message:
                        lines.append(f"[SYSTEM ALERT - max chars ({max_chars}) reached during file read]")
                    break
                else:
                    lines.append(line.rstrip("\n"))
                    chars_read += chars_to_add
            if current_line_number >= line_start + num_lines - 1:
                break

    return "\n".join(lines)


def append_to_text_file(self, filename: str, content: str):
    """
    Append to a text file.

    Args:
        filename (str): The name of the file to append to.
        content (str): Content to append to the file.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    with open(filename, "a", encoding="utf-8") as file:
        file.write(content + "\n")


def http_request(self, method: str, url: str, payload_json: Optional[str] = None):
    """
    Generates an HTTP request and returns the response.

    Args:
        method (str): The HTTP method (e.g., 'GET', 'POST').
        url (str): The URL for the request.
        payload_json (Optional[str]): A JSON string representing the request payload.

    Returns:
        dict: The response from the HTTP request.
    """
    try:
        headers = {"Content-Type": "application/json"}

        # For GET requests, ignore the payload
        if method.upper() == "GET":
            print(f"[HTTP] launching GET request to {url}")
            response = requests.get(url, headers=headers)
        else:
            # Validate and convert the payload for other types of requests
            if payload_json:
                payload = json_loads(payload_json)
            else:
                payload = {}
            print(f"[HTTP] launching {method} request to {url}, payload=\n{json_dumps(payload, indent=2)}")
            response = requests.request(method, url, json=payload, headers=headers)

        return {"status_code": response.status_code, "headers": dict(response.headers), "body": response.text}
    except Exception as e:
        return {"error": str(e)}
