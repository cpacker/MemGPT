import json


def extract_first_json(string):
    """Handles the case of two JSON objects back-to-back"""
    depth = 0
    start_index = None

    for i, char in enumerate(string):
        if char == "{":
            if depth == 0:
                start_index = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start_index is not None:
                try:
                    return json.loads(string[start_index : i + 1])
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"Matched closing bracket, but decode failed with error: {str(e)}")
    print("No valid JSON object found.")
    raise json.JSONDecodeError("Couldn't find starting bracket")


def add_missing_heartbeat(llm_json):
    """Manually insert heartbeat requests into messages that should have them

    Use the following heuristic:
      - if (function call is not send_message && prev message['role'] == user): insert heartbeat

    Basically, if MemGPT is calling a function (not send_message) immediately after the user sending a message,
    it probably is a retriever or insertion call, in which case we likely want to eventually reply with send_message

            "message" = {
            "role": "assistant",
            "content": ...,
            "function_call": {
                "name": ...
                "arguments": {
                    "arg1": val1,
                    ...
                }
            }
        }
    """
    raise NotImplementedError


def clean_json(raw_llm_output, messages=None, functions=None):
    """Try a bunch of hacks to parse the data coming out of the LLM"""

    try:
        data = json.loads(raw_llm_output)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw_llm_output + "}")
        except json.JSONDecodeError:
            try:
                data = extract_first_json(raw_llm_output + "}")
            except:
                raise

    return data
