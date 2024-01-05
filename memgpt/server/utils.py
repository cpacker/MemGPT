def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    if response.get("type") in ["agent_response_end", "agent_response_error", "command_response", "server_error"]:
        return True
    else:
        return False


def print_server_response(response):
    """Turn response json into a nice print"""
    if response["type"] == "agent_response_start":
        print("[agent.step start]")
    elif response["type"] == "agent_response_end":
        print("[agent.step end]")
    elif response["type"] == "agent_response":
        msg = response["message"]
        if response["message_type"] == "internal_monologue":
            print(f"[inner thoughts] {msg}")
        elif response["message_type"] == "assistant_message":
            print(f"{msg}")
        elif response["message_type"] == "function_message":
            pass
        else:
            print(response)
    else:
        print(response)


def shorten_key_middle(key_string, chars_each_side=3):
    """
    Shortens a key string by showing a specified number of characters on each side and adding an ellipsis in the middle.

    Args:
    key_string (str): The key string to be shortened.
    chars_each_side (int): The number of characters to show on each side of the ellipsis.

    Returns:
    str: The shortened key string with an ellipsis in the middle.
    """
    if not key_string:
        return key_string
    key_length = len(key_string)
    if key_length <= 2 * chars_each_side:
        return "..."  # Return ellipsis if the key is too short
    else:
        return key_string[:chars_each_side] + "..." + key_string[-chars_each_side:]
