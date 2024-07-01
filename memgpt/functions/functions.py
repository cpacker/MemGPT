import importlib
import inspect
import os
import sys
from textwrap import dedent  # remove indentation
from types import ModuleType

from memgpt.constants import CLI_WARNING_PREFIX, MEMGPT_DIR
from memgpt.functions.schema_generator import generate_schema

USER_FUNCTIONS_DIR = os.path.join(MEMGPT_DIR, "functions")

sys.path.append(USER_FUNCTIONS_DIR)


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
            f"'{file}' imports '{missing_package}', but '{missing_package}' is not installed locally - install python package '{missing_package}' to link functions from '{file}' to MemGPT.",
        )
    except SyntaxError as e:
        # Handle syntax errors in the module
        return False, f"{CLI_WARNING_PREFIX}skipped loading python file '{file}' due to a syntax error: {e}"
    except Exception as e:
        # Handle other general exceptions
        return False, f"{CLI_WARNING_PREFIX}skipped loading python file '{file}': {e}"

    return True, None


def write_function(module_name: str, function_name: str, function_code: str):
    """Write a function to a file in the user functions directory"""
    # Create the user functions directory if it doesn't exist
    if not os.path.exists(USER_FUNCTIONS_DIR):
        os.makedirs(USER_FUNCTIONS_DIR)

    # Write the function to a file
    file_path = os.path.join(USER_FUNCTIONS_DIR, f"{module_name}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(function_code)
    succ, error = validate_function(module_name, file_path)

    # raise error if function cannot be loaded
    if not succ:
        raise ValueError(error)
    return file_path


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
            f"'{file}' imports '{missing_package}', but '{missing_package}' is not installed locally - install python package '{missing_package}' to link functions from '{file}' to MemGPT."
        )
    # load all functions in the module
    function_dict = load_function_set(module)
    return function_dict
