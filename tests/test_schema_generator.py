from letta.functions.schema_generator import generate_schema


def send_message(self, message: str):
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None


def send_message_missing_types(self, message):
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None


def send_message_missing_docstring(self, message: str):
    return None


def test_schema_generator():
    # Check that a basic function schema converts correctly
    correct_schema = {
        "name": "send_message",
        "description": "Sends a message to the human user.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message contents. All unicode (including emojis) are supported."}},
            "required": ["message"],
        },
    }
    generated_schema = generate_schema(send_message)
    print(f"\n\nreference_schema={correct_schema}")
    print(f"\n\ngenerated_schema={generated_schema}")
    assert correct_schema == generated_schema

    # Check that missing types results in an error
    try:
        _ = generate_schema(send_message_missing_types)
        assert False
    except:
        pass

    # Check that missing docstring results in an error
    try:
        _ = generate_schema(send_message_missing_docstring)
        assert False
    except:
        pass
