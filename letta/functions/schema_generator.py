import inspect
from typing import Any, Dict, Optional, Type, Union, get_args, get_origin

from docstring_parser import parse
from pydantic import BaseModel
from pydantic.v1 import BaseModel as V1BaseModel


def is_optional(annotation):
    # Check if the annotation is a Union
    if getattr(annotation, "__origin__", None) is Union:
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
        list[str]: "array",
        # Add more mappings as needed
    }
    if py_type not in type_map:
        raise ValueError(f"Python type {py_type} has no corresponding JSON schema type")

    return type_map.get(py_type, "string")  # Default to "string" if type not in map


def pydantic_model_to_open_ai(model):
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}
    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (description := param.description):
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name]["description"] = description

    parameters["required"] = sorted(k for k, v in parameters["properties"].items() if "default" not in v)

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            raise

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


def generate_schema(function, name: Optional[str] = None, description: Optional[str] = None) -> dict:
    # Get the signature of the function
    sig = inspect.signature(function)

    # Parse the docstring
    docstring = parse(function.__doc__)

    # Prepare the schema dictionary
    schema = {
        "name": function.__name__ if name is None else name,
        "description": docstring.short_description if description is None else description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    # TODO: ensure that 'agent' keyword is reserved for `Agent` class

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

        if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel):
            schema["parameters"]["properties"][param.name] = pydantic_model_to_open_ai(param.annotation)
        else:
            # Add parameter details to the schema
            param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)
            schema["parameters"]["properties"][param.name] = {
                # "type": "string" if param.annotation == str else str(param.annotation),
                "type": type_to_json_schema_type(param.annotation) if param.annotation != inspect.Parameter.empty else "string",
                "description": param_doc.description,
            }
        if param.default == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param.name)

        if get_origin(param.annotation) is list:
            if get_args(param.annotation)[0] is str:
                schema["parameters"]["properties"][param.name]["items"] = {"type": "string"}

        if param.annotation == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param.name)

    # append the heartbeat
    # TODO: don't hard-code
    if function.__name__ not in ["send_message", "pause_heartbeats"]:
        schema["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.",
        }
        schema["parameters"]["required"].append("request_heartbeat")

    return schema


def generate_schema_from_args_schema_v1(
    args_schema: Type[V1BaseModel], name: Optional[str] = None, description: Optional[str] = None, append_heartbeat: bool = True
) -> Dict[str, Any]:
    properties = {}
    required = []
    for field_name, field in args_schema.__fields__.items():
        if field.type_ == str:
            field_type = "string"
        elif field.type_ == int:
            field_type = "integer"
        elif field.type_ == bool:
            field_type = "boolean"
        else:
            field_type = field.type_.__name__

        properties[field_name] = {
            "type": field_type,
            "description": field.field_info.description,
        }
        if field.required:
            required.append(field_name)

    function_call_json = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }

    if append_heartbeat:
        function_call_json["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.",
        }
        function_call_json["parameters"]["required"].append("request_heartbeat")

    return function_call_json


def generate_schema_from_args_schema_v2(
    args_schema: Type[BaseModel], name: Optional[str] = None, description: Optional[str] = None, append_heartbeat: bool = True
) -> Dict[str, Any]:
    properties = {}
    required = []
    for field_name, field in args_schema.model_fields.items():
        field_type_annotation = field.annotation
        if field_type_annotation == str:
            field_type = "string"
        elif field_type_annotation == int:
            field_type = "integer"
        elif field_type_annotation == bool:
            field_type = "boolean"
        else:
            field_type = field_type_annotation.__name__

        properties[field_name] = {
            "type": field_type,
            "description": field.description,
        }
        if field.is_required():
            required.append(field_name)

    function_call_json = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }

    if append_heartbeat:
        function_call_json["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.",
        }
        function_call_json["parameters"]["required"].append("request_heartbeat")

    return function_call_json
