import json

from memgpt.constants import JSON_ENSURE_ASCII

# Server -> client


def server_error(msg):
    """General server error"""
    return json.dumps(
        {
            "type": "server_error",
            "message": msg,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_command_response(status):
    return json.dumps(
        {
            "type": "command_response",
            "status": status,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_agent_response_error(msg):
    return json.dumps(
        {
            "type": "agent_response_error",
            "message": msg,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_agent_response_start():
    return json.dumps(
        {
            "type": "agent_response_start",
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_agent_response_end():
    return json.dumps(
        {
            "type": "agent_response_end",
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_agent_internal_monologue(msg):
    return json.dumps(
        {
            "type": "agent_response",
            "message_type": "internal_monologue",
            "message": msg,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_agent_assistant_message(msg):
    return json.dumps(
        {
            "type": "agent_response",
            "message_type": "assistant_message",
            "message": msg,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def server_agent_function_message(msg):
    return json.dumps(
        {
            "type": "agent_response",
            "message_type": "function_message",
            "message": msg,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


# Client -> server


def client_user_message(msg, agent_id=None):
    return json.dumps(
        {
            "type": "user_message",
            "message": msg,
            "agent_id": agent_id,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )


def client_command_create(config):
    return json.dumps(
        {
            "type": "command",
            "command": "create_agent",
            "config": config,
        },
        ensure_ascii=JSON_ENSURE_ASCII,
    )
