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


def repair_json_string(json_string):
    """
    This function repairs a JSON string where line feeds were accidentally added
    within string literals. The line feeds are replaced with the escaped line
    feed sequence '\\n'.
    """
    new_string = ""
    in_string = False
    escape = False

    for char in json_string:
        if char == '"' and not escape:
            in_string = not in_string
        if char == "\\" and not escape:
            escape = True
        else:
            escape = False
        if char == "\n" and in_string:
            new_string += "\\n"
        else:
            new_string += char

    return new_string


def repair_even_worse_json(json_string):
    """
    This function repairs a malformed JSON string where string literals are broken up and
    not properly enclosed in quotes. It aims to consolidate everything between 'message': and
    the two ending curly braces into one string for the 'message' field.
    """
    # State flags
    in_message = False
    in_string = False
    escape = False
    message_content = []

    # Storage for the new JSON
    new_json_parts = []

    # Iterating through each character
    for char in json_string:
        if char == '"' and not escape:
            in_string = not in_string
            if not in_message:
                # If we encounter a quote and are not in message, append normally
                new_json_parts.append(char)
        elif char == "\\" and not escape:
            escape = True
            new_json_parts.append(char)
        else:
            if escape:
                escape = False
            if in_message:
                if char == "}":
                    # Append the consolidated message and the closing characters then reset the flag
                    new_json_parts.append('"{}"'.format("".join(message_content).replace("\n", " ")))
                    new_json_parts.append(char)
                    in_message = False
                elif in_string or char.isalnum() or char.isspace() or char in ".',;:!":
                    # Collect the message content, excluding structural characters
                    message_content.append(char)
            else:
                # If we're not in message mode, append character to the output as is
                new_json_parts.append(char)
                if '"message":' in "".join(new_json_parts[-10:]):
                    # If we detect "message": pattern, switch to message mode
                    in_message = True
                    message_content = []

    # Joining everything to form the new JSON
    repaired_json = "".join(new_json_parts)
    return repaired_json


def clean_json(raw_llm_output, messages=None, functions=None):
    from memgpt.utils import printd

    """Try a bunch of hacks to parse the data coming out of the LLM"""
    try:
        # printd("clean json runs:", raw_llm_output)
        data = json.loads(raw_llm_output)
    except json.JSONDecodeError:
        try:
            printd("trying adding }")
            data = json.loads(raw_llm_output + "}")
        except json.JSONDecodeError:
            try:
                printd("trying adding }}")
                data = json.loads(raw_llm_output + "}}")
            except json.JSONDecodeError:
                try:
                    printd('trying adding "}}')
                    data = json.loads(raw_llm_output + '"}}')
                except json.JSONDecodeError:
                    try:
                        repaired = repair_json_string(raw_llm_output)
                        printd("trying repair_json_string:", repaired)
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        try:
                            repaired = repair_even_worse_json(raw_llm_output)
                            printd("trying repair_even_worse_json:", repaired)
                            data = json.loads(repaired)
                        except json.JSONDecodeError:
                            try:
                                printd("trying first_json")
                                data = extract_first_json(raw_llm_output + "}}")
                            except:
                                raise
    return data
