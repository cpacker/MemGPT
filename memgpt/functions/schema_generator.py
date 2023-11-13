import inspect
import typing
from typing import get_args

from docstring_parser import parse

from memgpt.constants import FUNCTION_PARAM_NAME_REQ_HEARTBEAT, FUNCTION_PARAM_TYPE_REQ_HEARTBEAT, FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT

NO_HEARTBEAT_FUNCTIONS = ["send_message", "pause_heartbeats"]


def is_optional(annotation):
    # Check if the annotation is a Union
    if getattr(annotation, "__origin__", None) is typing.Union:
        # Check if None is one of the options in the Union
        return type(None) in annotation.__args__
    return False


def optional_length(annotation):
    if is_optional(annotation):
        # Subtract 1 to account for NoneType
        return len(annotation.__args__) - 1
    else:
        raise ValueError("The annotation is not an Optional type")


def type_to_json_schema_type(py_type):
    """
    Maps a Python type to a JSON schema type.
    Specifically handles typing.Optional and common Python types.
    """
    # if get_origin(py_type) is typing.Optional:
    if is_optional(py_type):
        # Assert that Optional has only one type argument
        type_args = get_args(py_type)
        assert optional_length(py_type) == 1, f"Optional type must have exactly one type argument, but got {py_type}"

        # Extract and map the inner type
        return type_to_json_schema_type(type_args[0])

    # Mapping of Python types to JSON schema types
    type_map = {
        int: "integer",
        str: "string",
        bool: "boolean",
        float: "number",
        # Add more mappings as needed
    }
    if py_type not in type_map:
        raise ValueError(f"Python type {py_type} has no corresponding JSON schema type")

    return type_map.get(py_type, "string")  # Default to "string" if type not in map


def generate_schema(function):
    # Get the signature of the function
    sig = inspect.signature(function)

    # Parse the docstring
    docstring = parse(function.__doc__)

    # Prepare the schema dictionary
    schema = {
        "name": function.__name__,
        "description": docstring.short_description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    for param in sig.parameters.values():
        # Exclude 'self' parameter
        if param.name == "self":
            continue

        # Assert that the parameter has a type annotation
        if param.annotation == inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a type annotation")

        # Find the parameter's description in the docstring
        param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)

        # Assert that the parameter has a description
        if not param_doc or not param_doc.description:
            raise ValueError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a description in the docstring")

        # Add parameter details to the schema
        param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)
        schema["parameters"]["properties"][param.name] = {
            # "type": "string" if param.annotation == str else str(param.annotation),
            "type": type_to_json_schema_type(param.annotation) if param.annotation != inspect.Parameter.empty else "string",
            "description": param_doc.description,
        }
        if param.default == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param.name)

    # append the heartbeat
    if function.__name__ not in NO_HEARTBEAT_FUNCTIONS:
        schema["parameters"]["properties"][FUNCTION_PARAM_NAME_REQ_HEARTBEAT] = {
            "type": FUNCTION_PARAM_TYPE_REQ_HEARTBEAT,
            "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
        }
        schema["parameters"]["required"].append(FUNCTION_PARAM_NAME_REQ_HEARTBEAT)

    return schema
