from letta.utils import json_dumps

# Server -> client


def server_error(msg):
    """General server error"""
    return json_dumps(
        {
            "type": "server_error",
            "message": msg,
        }
    )


def server_command_response(status):
    return json_dumps(
        {
            "type": "command_response",
            "status": status,
        }
    )


def server_agent_response_error(msg):
    return json_dumps(
        {
            "type": "agent_response_error",
            "message": msg,
        }
    )


def server_agent_response_start():
    return json_dumps(
        {
            "type": "agent_response_start",
        }
    )


def server_agent_response_end():
    return json_dumps(
        {
            "type": "agent_response_end",
        }
    )


def server_agent_internal_monologue(msg):
    return json_dumps(
        {
            "type": "agent_response",
            "message_type": "internal_monologue",
            "message": msg,
        }
    )


def server_agent_assistant_message(msg):
    return json_dumps(
        {
            "type": "agent_response",
            "message_type": "assistant_message",
            "message": msg,
        }
    )


def server_agent_function_message(msg):
    return json_dumps(
        {
            "type": "agent_response",
            "message_type": "function_message",
            "message": msg,
        }
    )


# Client -> server


def client_user_message(msg, agent_id=None):
    return json_dumps(
        {
            "type": "user_message",
            "message": msg,
            "agent_id": agent_id,
        }
    )


def client_command_create(config):
    return json_dumps(
        {
            "type": "command",
            "command": "create_agent",
            "config": config,
        }
    )
