import json

# Server -> client


def server_error(msg):
    """General server error"""
    return json.dumps(
        {
            "type": "server_error",
            "message": msg,
        }
    )


def server_command_response(status):
    return json.dumps(
        {
            "type": "command_response",
            "status": status,
        }
    )


def server_agent_response_error(msg):
    return json.dumps(
        {
            "type": "agent_response_error",
            "message": msg,
        }
    )


def server_agent_response_start():
    return json.dumps(
        {
            "type": "agent_response_start",
        }
    )


def server_agent_response_end():
    return json.dumps(
        {
            "type": "agent_response_end",
        }
    )


def server_agent_internal_monologue(msg):
    return json.dumps(
        {
            "type": "agent_response",
            "message_type": "internal_monologue",
            "message": msg,
        }
    )


def server_agent_assistant_message(msg):
    return json.dumps(
        {
            "type": "agent_response",
            "message_type": "assistant_message",
            "message": msg,
        }
    )


def server_agent_function_message(msg):
    return json.dumps(
        {
            "type": "agent_response",
            "message_type": "function_message",
            "message": msg,
        }
    )


# Client -> server


def client_user_message(msg, agent_name=None):
    return json.dumps(
        {
            "type": "user_message",
            "message": msg,
            "agent_name": agent_name,
        }
    )


def client_command_create(config):
    return json.dumps(
        {
            "type": "command",
            "command": "create_agent",
            "config": config,
        }
    )


def client_command_load(agent_name):
    return json.dumps(
        {
            "type": "command",
            "command": "load_agent",
            "name": agent_name,
        }
    )
