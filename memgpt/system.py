import uuid
import json

from .utils import get_local_time
from .constants import (
    MESSAGE_SUMMARY_WARNING_STR,
    JSON_ENSURE_ASCII,
)


def get_initial_boot_messages():
    tool_call_id = str(uuid.uuid4())
    messages = [
        # first message includes both inner monologue and function call to send_message
        {
            "role": "assistant",
            "content": "Bootup sequence complete. Persona activated. Testing messaging functionality.",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "arguments": '{\n  "message": "' + "I am ready to chat" + '"\n}',
                    },
                }
            ],
        },
        # obligatory function return message
        {
            # "role": "function",
            "role": "tool",
            "name": "send_message",  # NOTE: technically not up to spec, this is old functions style
            "content": package_function_response(True, None),
            "tool_call_id": tool_call_id,
        },
    ]

    return messages


def get_heartbeat(reason="Automated timer", include_location=False, location_name="San Francisco, CA, USA"):
    # Package the message with time and location
    formatted_time = get_local_time()
    packaged_message = {
        "type": "heartbeat",
        "reason": reason,
        "time": formatted_time,
    }

    if include_location:
        packaged_message["location"] = location_name

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def get_login_event(last_login="Never (first login)", include_location=False, location_name="San Francisco, CA, USA"):
    # Package the message with time and location
    formatted_time = get_local_time()
    packaged_message = {
        "type": "login",
        "last_login": last_login,
        "time": formatted_time,
    }

    if include_location:
        packaged_message["location"] = location_name

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def package_user_message(user_message, time=None, include_location=False, location_name="San Francisco, CA, USA", name=None):
    # Package the message with time and location
    formatted_time = time if time else get_local_time()
    packaged_message = {
        "type": "user_message",
        "message": user_message,
        "time": formatted_time,
    }

    if include_location:
        packaged_message["location"] = location_name

    if name:
        packaged_message["name"] = name

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def package_function_response(was_success, response_string, timestamp=None):
    formatted_time = get_local_time() if timestamp is None else timestamp
    packaged_message = {
        "status": "OK" if was_success else "Failed",
        "message": response_string,
        "time": formatted_time,
    }

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def package_system_message(system_message, message_type="system_alert", time=None):
    formatted_time = time if time else get_local_time()
    packaged_message = {
        "type": message_type,
        "message": system_message,
        "time": formatted_time,
    }

    return json.dumps(packaged_message)


def package_summarize_message(summary, summary_length, timestamp=None):
    context_message = (
        f"Note: prior messages have been hidden from view due to conversation memory constraints.\n"
        + f"The following is a summary of the previous {summary_length} messages:\n {summary}"
    )

    formatted_time = get_local_time() if timestamp is None else timestamp
    packaged_message = {
        "type": "system_alert",
        "message": context_message,
        "time": formatted_time,
    }

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def get_token_limit_warning():
    formatted_time = get_local_time()
    packaged_message = {
        "type": "system_alert",
        "message": MESSAGE_SUMMARY_WARNING_STR,
        "time": formatted_time,
    }

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)
