import json

from memgpt.local_llm.function_parser import patch_function
import memgpt.system as system


EXAMPLE_FUNCTION_CALL_SEND_MESSAGE = {
    "message_history": [
        {"role": "user", "content": system.package_user_message("hello")},
    ],
    # "new_message": {
    #     "role": "function",
    #     "name": "send_message",
    #     "content": system.package_function_response(was_success=True, response_string="None"),
    # },
    "new_message": {
        "role": "assistant",
        "content": "I'll send a message.",
        "function_call": {
            "name": "send_message",
            "arguments": "null",
        },
    },
}

EXAMPLE_FUNCTION_CALL_CORE_MEMORY_APPEND_MISSING = {
    "message_history": [
        {"role": "user", "content": system.package_user_message("hello")},
    ],
    "new_message": {
        "role": "assistant",
        "content": "I'll append to memory.",
        "function_call": {
            "name": "core_memory_append",
            "arguments": json.dumps({"content": "new_stuff"}, ensure_ascii=False),
        },
    },
}


def test_function_parsers():
    """Try various broken JSON and check that the parsers can fix it"""

    og_message = EXAMPLE_FUNCTION_CALL_SEND_MESSAGE["new_message"]
    corrected_message = patch_function(**EXAMPLE_FUNCTION_CALL_SEND_MESSAGE)
    assert corrected_message == og_message, f"Uncorrected:\n{og_message}\nCorrected:\n{corrected_message}"

    og_message = EXAMPLE_FUNCTION_CALL_CORE_MEMORY_APPEND_MISSING["new_message"].copy()
    corrected_message = patch_function(**EXAMPLE_FUNCTION_CALL_CORE_MEMORY_APPEND_MISSING)
    assert corrected_message != og_message, f"Uncorrected:\n{og_message}\nCorrected:\n{corrected_message}"
