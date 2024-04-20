import json
import re

from memgpt.constants import JSON_LOADS_STRICT
from memgpt.errors import LLMJSONParsingError


def clean_json_string_extra_backslash(s):
    """Clean extra backslashes out from stringified JSON

    NOTE: Google AI Gemini API likes to include these
    """
    # Strip slashes that are used to escape single quotes and other backslashes
    # Use json.loads to parse it correctly
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")
    return s


def replace_escaped_underscores(string: str):
    """Handles the case of escaped underscores, e.g.:

    {
      "function":"send\_message",
      "params": {
        "inner\_thoughts": "User is asking for information about themselves. Retrieving data from core memory.",
        "message": "I know that you are Chad. Is there something specific you would like to know or talk about regarding yourself?"
    """
    return string.replace("\_", "_")


def extract_first_json(string: str):
    """Handles the case of two JSON objects back-to-back"""
    from memgpt.utils import printd

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
                    return json.loads(string[start_index : i + 1], strict=JSON_LOADS_STRICT)
                except json.JSONDecodeError as e:
                    raise LLMJSONParsingError(f"Matched closing bracket, but decode failed with error: {str(e)}")
    printd("No valid JSON object found.")
    raise LLMJSONParsingError("Couldn't find starting bracket")


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


def clean_and_interpret_send_message_json(json_string):
    # If normal parsing fails, attempt to clean and extract manually
    cleaned_json_string = re.sub(r"[^\x00-\x7F]+", "", json_string)  # Remove non-ASCII characters
    function_match = re.search(r'"function":\s*"send_message"', cleaned_json_string)
    inner_thoughts_match = re.search(r'"inner_thoughts":\s*"([^"]+)"', cleaned_json_string)
    message_match = re.search(r'"message":\s*"([^"]+)"', cleaned_json_string)

    if function_match and inner_thoughts_match and message_match:
        return {
            "function": "send_message",
            "params": {
                "inner_thoughts": inner_thoughts_match.group(1),
                "message": message_match.group(1),
            },
        }
    else:
        raise LLMJSONParsingError(f"Couldn't manually extract send_message pattern from:\n{json_string}")


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

    strategies = [
        lambda output: json.loads(output, strict=JSON_LOADS_STRICT),
        lambda output: json.loads(output + "}", strict=JSON_LOADS_STRICT),
        lambda output: json.loads(output + "}}", strict=JSON_LOADS_STRICT),
        lambda output: json.loads(output + '"}}', strict=JSON_LOADS_STRICT),
        # with strip and strip comma
        lambda output: json.loads(output.strip().rstrip(",") + "}", strict=JSON_LOADS_STRICT),
        lambda output: json.loads(output.strip().rstrip(",") + "}}", strict=JSON_LOADS_STRICT),
        lambda output: json.loads(output.strip().rstrip(",") + '"}}', strict=JSON_LOADS_STRICT),
        # more complex patchers
        lambda output: json.loads(repair_json_string(output), strict=JSON_LOADS_STRICT),
        lambda output: json.loads(repair_even_worse_json(output), strict=JSON_LOADS_STRICT),
        lambda output: extract_first_json(output + "}}"),
        lambda output: clean_and_interpret_send_message_json(output),
        # replace underscores
        lambda output: json.loads(replace_escaped_underscores(output), strict=JSON_LOADS_STRICT),
        lambda output: extract_first_json(replace_escaped_underscores(output) + "}}"),
    ]

    for strategy in strategies:
        try:
            printd(f"Trying strategy: {strategy.__name__}")
            return strategy(raw_llm_output)
        except (json.JSONDecodeError, LLMJSONParsingError) as e:
            printd(f"Strategy {strategy.__name__} failed with error: {e}")

    raise LLMJSONParsingError(f"Failed to decode valid MemGPT JSON from LLM output:\n=====\n{raw_llm_output}\n=====")
