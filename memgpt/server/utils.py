def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    return response.get("type") == "agent_response_end"


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
