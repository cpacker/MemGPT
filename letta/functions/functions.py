import importlib
import inspect
import os
from textwrap import dedent  # remove indentation
from types import ModuleType
from typing import Dict, List, Optional

from letta.constants import CLI_WARNING_PREFIX
from letta.errors import LettaToolCreateError
from letta.functions.schema_generator import generate_schema


def derive_openai_json_schema(source_code: str, name: Optional[str] = None) -> dict:
    """Derives the OpenAI JSON schema for a given function source code.

    First, attempts to execute the source code in a custom environment with only the necessary imports.
    Then, it generates the schema from the function's docstring and signature.
    """
    try:
        # Define a custom environment with necessary imports
        env = {
            "Optional": Optional,
            "List": List,
            "Dict": Dict,
            # To support Pydantic models
            # "BaseModel": BaseModel,
            # "Field": Field,
        }
        env.update(globals())

        # print("About to execute source code...")
        exec(source_code, env)
        # print("Source code executed successfully")

        functions = [f for f in env if callable(env[f]) and not f.startswith("__")]
        if not functions:
            raise LettaToolCreateError("No callable functions found in source code")

        # print(f"Found functions: {functions}")
        func = env[functions[-1]]

        if not hasattr(func, "__doc__") or not func.__doc__:
            raise LettaToolCreateError(f"Function {func.__name__} missing docstring")

        # print("About to generate schema...")
        try:
            schema = generate_schema(func, name=name)
            # print("Schema generated successfully")
            return schema
        except TypeError as e:
            raise LettaToolCreateError(f"Type error in schema generation: {str(e)}")
        except ValueError as e:
            raise LettaToolCreateError(f"Value error in schema generation: {str(e)}")
        except Exception as e:
            raise LettaToolCreateError(f"Unexpected error in schema generation: {str(e)}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LettaToolCreateError(f"Schema generation failed: {str(e)}") from e


def parse_source_code(func) -> str:
    """Parse the source code of a function and remove indendation"""
    source_code = dedent(inspect.getsource(func))
    return source_code


def load_function_set(module: ModuleType) -> dict:
    """Load the functions and generate schema for them, given a module object"""
    function_dict = {}

    for attr_name in dir(module):
        # Get the attribute
        attr = getattr(module, attr_name)

        # Check if it's a callable function and not a built-in or special method
        if inspect.isfunction(attr) and attr.__module__ == module.__name__:
            if attr_name in function_dict:
                raise ValueError(f"Found a duplicate of function name '{attr_name}'")

            generated_schema = generate_schema(attr)
            function_dict[attr_name] = {
                "module": inspect.getsource(module),
                "python_function": attr,
                "json_schema": generated_schema,
            }

    if len(function_dict) == 0:
        raise ValueError(f"No functions found in module {module}")
    return function_dict


def validate_function(module_name, module_full_path):
    try:
        file = os.path.basename(module_full_path)
        spec = importlib.util.spec_from_file_location(module_name, module_full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        # Handle missing module imports
        missing_package = str(e).split("'")[1]  # Extract the name of the missing package
        print(f"{CLI_WARNING_PREFIX}skipped loading python file '{module_full_path}'!")
        return (
            False,
            f"'{file}' imports '{missing_package}', but '{missing_package}' is not installed locally - install python package '{missing_package}' to link functions from '{file}' to Letta.",
        )
    except SyntaxError as e:
        # Handle syntax errors in the module
        return False, f"{CLI_WARNING_PREFIX}skipped loading python file '{file}' due to a syntax error: {e}"
    except Exception as e:
        # Handle other general exceptions
        return False, f"{CLI_WARNING_PREFIX}skipped loading python file '{file}': {e}"

    return True, None


def load_function_file(filepath: str) -> dict:
    file = os.path.basename(filepath)
    module_name = file[:-3]  # Remove '.py' from filename
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        # Handle missing module imports
        missing_package = str(e).split("'")[1]  # Extract the name of the missing package
        print(f"{CLI_WARNING_PREFIX}skipped loading python file '{filepath}'!")
        print(
            f"'{file}' imports '{missing_package}', but '{missing_package}' is not installed locally - install python package '{missing_package}' to link functions from '{file}' to Letta."
        )
    # load all functions in the module
    function_dict = load_function_set(module)
    return function_dict
