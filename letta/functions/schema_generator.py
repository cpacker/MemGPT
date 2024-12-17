import inspect
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

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


def type_to_json_schema_type(py_type) -> dict:
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

    # Handle Union types (except Optional which is handled above)
    if get_origin(py_type) is Union:
        # TODO support mapping Unions to anyOf
        raise NotImplementedError("General Union types are not yet supported")

    # Handle array types
    origin = get_origin(py_type)
    if py_type == list or origin in (list, List):
        args = get_args(py_type)

        if args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):
            # If it's a list of Pydantic models, return an array with the model schema as items
            return {
                "type": "array",
                "items": pydantic_model_to_json_schema(args[0]),
            }

        # Otherwise, recursively call the basic type checker
        return {
            "type": "array",
            # get the type of the items in the list
            "items": type_to_json_schema_type(args[0]),
        }

    # Handle object types
    if py_type == dict or origin in (dict, Dict):
        args = get_args(py_type)
        if not args:
            # Generic dict without type arguments
            return {
                "type": "object",
                # "properties": {}
            }
        else:
            raise ValueError(
                f"Dictionary types {py_type} with nested type arguments are not supported (consider using a Pydantic model instead)"
            )

        # NOTE: the below code works for generic JSON schema parsing, but there's a problem with the key inference
        #       when it comes to OpenAI function schema generation so it doesn't make sense to allow for dict[str, Any] type hints
        # key_type, value_type = args

        # # Ensure dict keys are strings
        # # Otherwise there's no JSON schema equivalent
        # if key_type != str:
        #     raise ValueError("Dictionary keys must be strings for OpenAI function schema compatibility")

        # # Handle value type to determine property schema
        # value_schema = {}
        # if inspect.isclass(value_type) and issubclass(value_type, BaseModel):
        #     value_schema = pydantic_model_to_json_schema(value_type)
        # else:
        #     value_schema = type_to_json_schema_type(value_type)

        # # NOTE: the problem lies here - the key is always "key_placeholder"
        # return {"type": "object", "properties": {"key_placeholder": value_schema}}

    # Handle direct Pydantic models
    if inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        return pydantic_model_to_json_schema(py_type)

    # Mapping of Python types to JSON schema types
    type_map = {
        # Basic types
        # Optional, Union, and collections are handled above ^
        int: "integer",
        str: "string",
        bool: "boolean",
        float: "number",
        None: "null",
    }
    if py_type not in type_map:
        raise ValueError(f"Python type {py_type} has no corresponding JSON schema type - full map: {type_map}")
    else:
        return {"type": type_map[py_type]}


def pydantic_model_to_open_ai(model: Type[BaseModel]) -> dict:
    """
    Converts a Pydantic model as a singular arg to a JSON schema object for use in OpenAI function calling.
    """
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
            raise ValueError(f"No description found in docstring or description field (model: {model}, docstring: {docstring})")

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


def pydantic_model_to_json_schema(model: Type[BaseModel]) -> dict:
    """
    Converts a Pydantic model (as an arg that already is annotated) to a JSON schema object for use in OpenAI function calling.

    An example of a Pydantic model as an arg:

    class Step(BaseModel):
        name: str = Field(
            ...,
            description="Name of the step.",
        )
        key: str = Field(
            ...,
            description="Unique identifier for the step.",
        )
        description: str = Field(
            ...,
            description="An exhaustic description of what this step is trying to achieve and accomplish.",
        )

    def create_task_plan(steps: list[Step]):
        '''
        Creates a task plan for the current task.

        Args:
            steps: List of steps to add to the task plan.
        ...

    Should result in:
    {
      "name": "create_task_plan",
      "description": "Creates a task plan for the current task.",
      "parameters": {
        "type": "object",
        "properties": {
          "steps": {  # <= this is the name of the arg
            "type": "object",
            "description": "List of steps to add to the task plan.",
            "properties": {
              "name": {
                "type": "str",
                "description": "Name of the step.",
              },
              "key": {
                "type": "str",
                "description": "Unique identifier for the step.",
              },
              "description": {
                "type": "str",
                "description": "An exhaustic description of what this step is trying to achieve and accomplish.",
              },
            },
            "required": ["name", "key", "description"],
          }
        },
        "required": ["steps"],
      }
    }

    Specifically, the result of pydantic_model_to_json_schema(steps) (where `steps` is an instance of BaseModel) is:
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "str",
                "description": "Name of the step."
            },
            "key": {
                "type": "str",
                "description": "Unique identifier for the step."
            },
            "description": {
                "type": "str",
                "description": "An exhaustic description of what this step is trying to achieve and accomplish."
            },
        },
        "required": ["name", "key", "description"],
    }
    """
    schema = model.model_json_schema()

    def clean_property(prop: dict) -> dict:
        """Clean up a property schema to match desired format"""

        if "description" not in prop:
            raise ValueError(f"Property {prop} lacks a 'description' key")

        return {
            "type": "string" if prop["type"] == "string" else prop["type"],
            "description": prop["description"],
        }

    def resolve_ref(ref: str, schema: dict) -> dict:
        """Resolve a $ref reference in the schema"""
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unexpected reference format: {ref}")

        model_name = ref.split("/")[-1]
        if model_name not in schema.get("$defs", {}):
            raise ValueError(f"Reference {model_name} not found in schema definitions")

        return schema["$defs"][model_name]

    def clean_schema(schema_part: dict, full_schema: dict) -> dict:
        """Clean up a schema part, handling references and nested structures"""
        # Handle $ref
        if "$ref" in schema_part:
            schema_part = resolve_ref(schema_part["$ref"], full_schema)

        if "type" not in schema_part:
            raise ValueError(f"Schema part lacks a 'type' key: {schema_part}")

        # Handle array type
        if schema_part["type"] == "array":
            items_schema = schema_part["items"]
            if "$ref" in items_schema:
                items_schema = resolve_ref(items_schema["$ref"], full_schema)
            return {"type": "array", "items": clean_schema(items_schema, full_schema), "description": schema_part.get("description", "")}

        # Handle object type
        if schema_part["type"] == "object":
            if "properties" not in schema_part:
                raise ValueError(f"Object schema lacks 'properties' key: {schema_part}")

            properties = {}
            for name, prop in schema_part["properties"].items():
                if "items" in prop:  # Handle arrays
                    if "description" not in prop:
                        raise ValueError(f"Property {prop} lacks a 'description' key")
                    properties[name] = {
                        "type": "array",
                        "items": clean_schema(prop["items"], full_schema),
                        "description": prop["description"],
                    }
                else:
                    properties[name] = clean_property(prop)

            pydantic_model_schema_dict = {
                "type": "object",
                "properties": properties,
                "required": schema_part.get("required", []),
            }
            if "description" in schema_part:
                pydantic_model_schema_dict["description"] = schema_part["description"]

            return pydantic_model_schema_dict

        # Handle primitive types
        return clean_property(schema_part)

    return clean_schema(schema_part=schema, full_schema=schema)


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
        # TODO: eventually remove this (only applies to BASE_TOOLS)
        if param.name in ["self", "agent_state"]:  # Add agent_manager to excluded
            continue

        # Assert that the parameter has a type annotation
        if param.annotation == inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a type annotation")

        # Find the parameter's description in the docstring
        param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)

        # Assert that the parameter has a description
        if not param_doc or not param_doc.description:
            raise ValueError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a description in the docstring")

        # If the parameter is a pydantic model, we need to unpack the Pydantic model type into a JSON schema object
        # if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel):
        if (
            (inspect.isclass(param.annotation) or inspect.isclass(get_origin(param.annotation) or param.annotation))
            and not get_origin(param.annotation)
            and issubclass(param.annotation, BaseModel)
        ):
            # print("Generating schema for pydantic model:", param.annotation)
            # Extract the properties from the pydantic model
            schema["parameters"]["properties"][param.name] = pydantic_model_to_json_schema(param.annotation)
            schema["parameters"]["properties"][param.name]["description"] = param_doc.description

        # Otherwise, we convert the Python typing to JSON schema types
        # NOTE: important - if a dict or list, the internal type can be a Pydantic model itself
        #                   however in that
        else:
            # print("Generating schema for non-pydantic model:", param.annotation)
            # Grab the description for the parameter from the extended docstring
            # If it doesn't exist, we should raise an error
            param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)
            if not param_doc:
                raise ValueError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a description in the docstring")
            elif not isinstance(param_doc.description, str):
                raise ValueError(
                    f"Parameter '{param.name}' in function '{function.__name__}' has a description in the docstring that is not a string (type: {type(param_doc.description)})"
                )
            else:
                # If it's a string or a basic type, then all you need is: (1) type, (2) description
                # If it's a more complex type, then you also need either:
                # - for array, you need "items", each of which has "type"
                # - for a dict, you need "properties", which has keys which each have "type"
                if param.annotation != inspect.Parameter.empty:
                    param_generated_schema = type_to_json_schema_type(param.annotation)
                else:
                    # TODO why are we inferring here?
                    param_generated_schema = {"type": "string"}

                # Add in the description
                param_generated_schema["description"] = param_doc.description

                # Add the schema to the function arg key
                schema["parameters"]["properties"][param.name] = param_generated_schema

        # If the parameter doesn't have a default value, it is required (so we need to add it to the required list)
        if param.default == inspect.Parameter.empty and not is_optional(param.annotation):
            schema["parameters"]["required"].append(param.name)

        # TODO what's going on here?
        # If the parameter is a list of strings we need to hard cast to "string" instead of `str`
        if get_origin(param.annotation) is list:
            if get_args(param.annotation)[0] is str:
                schema["parameters"]["properties"][param.name]["items"] = {"type": "string"}

        # TODO is this not duplicating the other append directly above?
        if param.annotation == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param.name)

    # append the heartbeat
    # TODO: don't hard-code
    # TODO: if terminal, don't include this
    if function.__name__ not in ["send_message"]:
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
