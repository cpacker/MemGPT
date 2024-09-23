import copy
import json

from letta.utils import json_dumps, json_loads

NO_HEARTBEAT_FUNCS = ["send_message", "pause_heartbeats"]


def insert_heartbeat(message):
    # message_copy = message.copy()
    message_copy = copy.deepcopy(message)

    if message_copy.get("function_call"):
        # function_name = message.get("function_call").get("name")
        params = message_copy.get("function_call").get("arguments")
        params = json_loads(params)
        params["request_heartbeat"] = True
        message_copy["function_call"]["arguments"] = json_dumps(params)

    elif message_copy.get("tool_call"):
        # function_name = message.get("tool_calls")[0].get("function").get("name")
        params = message_copy.get("tool_calls")[0].get("function").get("arguments")
        params = json_loads(params)
        params["request_heartbeat"] = True
        message_copy["tools_calls"][0]["function"]["arguments"] = json_dumps(params)

    return message_copy


def heartbeat_correction(message_history, new_message):
    """Add heartbeats where we think the agent forgot to add them themselves

    If the last message in the stack is a user message and the new message is an assistant func call, fix the heartbeat

    See: https://github.com/cpacker/Letta/issues/601
    """
    if len(message_history) < 1:
        return None

    last_message_was_user = False
    if message_history[-1]["role"] == "user":
        try:
            content = json_loads(message_history[-1]["content"])
        except json.JSONDecodeError:
            return None
        # Check if it's a user message or system message
        if content["type"] == "user_message":
            last_message_was_user = True

    new_message_is_heartbeat_function = False
    if new_message["role"] == "assistant":
        if new_message.get("function_call") or new_message.get("tool_calls"):
            if new_message.get("function_call"):
                function_name = new_message.get("function_call").get("name")
            elif new_message.get("tool_calls"):
                function_name = new_message.get("tool_calls")[0].get("function").get("name")
            if function_name not in NO_HEARTBEAT_FUNCS:
                new_message_is_heartbeat_function = True

    if last_message_was_user and new_message_is_heartbeat_function:
        return insert_heartbeat(new_message)
    else:
        return None


def patch_function(message_history, new_message):
    corrected_output = heartbeat_correction(message_history=message_history, new_message=new_message)
    return corrected_output if corrected_output is not None else new_message
