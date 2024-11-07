import importlib
import inspect
import os
from textwrap import dedent  # remove indentation
from types import ModuleType
from typing import Optional

from letta.constants import CLI_WARNING_PREFIX
from letta.functions.schema_generator import generate_schema


def derive_openai_json_schema(source_code: str, name: Optional[str] = None) -> dict:
    # auto-generate openai schema
    try:
        # Define a custom environment with necessary imports
        env = {
            "Optional": Optional,  # Add any other required imports here
        }

        env.update(globals())
        exec(source_code, env)

        # get available functions
        functions = [f for f in env if callable(env[f])]

        # TODO: not sure if this always works
        func = env[functions[-1]]
        json_schema = generate_schema(func, name=name)
        return json_schema
    except Exception as e:
        raise RuntimeError(f"Failed to execute source code: {e}")


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
