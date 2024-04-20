import inspect

import memgpt.functions.function_sets.base as base_functions
import memgpt.functions.function_sets.extras as extras_functions
from memgpt.functions.schema_generator import generate_schema
from memgpt.prompts.gpt_functions import FUNCTIONS_CHAINING


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


def test_schema_generator_with_old_function_set():
    # Try all the base functions first
    for attr_name in dir(base_functions):
        # Get the attribute
        attr = getattr(base_functions, attr_name)

        # Check if it's a callable function and not a built-in or special method
        if inspect.isfunction(attr):
            # Here, 'func' is each function in base_functions
            # You can now call the function or do something with it
            print("Function name:", attr)
            # Example function call (if the function takes no arguments)
            # result = func()
            function_name = str(attr_name)
            real_schema = FUNCTIONS_CHAINING[function_name]
            generated_schema = generate_schema(attr)
            print(f"\n\nreference_schema={real_schema}")
            print(f"\n\ngenerated_schema={generated_schema}")
            assert real_schema == generated_schema

    # Then try all the extras functions
    for attr_name in dir(extras_functions):
        # Get the attribute
        attr = getattr(extras_functions, attr_name)

        # Check if it's a callable function and not a built-in or special method
        if inspect.isfunction(attr):
            if attr_name == "create":
                continue
            # Here, 'func' is each function in base_functions
            # You can now call the function or do something with it
            print("Function name:", attr)
            # Example function call (if the function takes no arguments)
            # result = func()
            function_name = str(attr_name)
            real_schema = FUNCTIONS_CHAINING[function_name]
            generated_schema = generate_schema(attr)
            print(f"\n\nreference_schema={real_schema}")
            print(f"\n\ngenerated_schema={generated_schema}")
            assert real_schema == generated_schema
