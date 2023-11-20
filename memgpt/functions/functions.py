import importlib
import inspect
import os
import sys


from memgpt.functions.schema_generator import generate_schema
from memgpt.constants import MEMGPT_DIR

sys.path.append(os.path.join(MEMGPT_DIR, "functions"))


def load_function_set(module):
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
                "python_function": attr,
                "json_schema": generated_schema,
            }

    if len(function_dict) == 0:
        raise ValueError(f"No functions found in module {module}")
    return function_dict


def load_all_function_sets(merge=True):
    # functions/examples/*.py
    scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    function_sets_dir = os.path.join(scripts_dir, "function_sets")  # Path to the function_sets directory
    # List all .py files in the directory (excluding __init__.py)
    example_module_files = [f for f in os.listdir(function_sets_dir) if f.endswith(".py") and f != "__init__.py"]

    # ~/.memgpt/functions/*.py
    user_scripts_dir = os.path.join(MEMGPT_DIR, "functions")
    # create if missing
    if not os.path.exists(user_scripts_dir):
        os.makedirs(user_scripts_dir)
    user_module_files = [f for f in os.listdir(user_scripts_dir) if f.endswith(".py") and f != "__init__.py"]

    # combine them both (pull from both examples and user-provided)
    # all_module_files = example_module_files + user_module_files

    # Add user_scripts_dir to sys.path
    if user_scripts_dir not in sys.path:
        sys.path.append(user_scripts_dir)

    schemas_and_functions = {}
    for dir_path, module_files in [(function_sets_dir, example_module_files), (user_scripts_dir, user_module_files)]:
        for file in module_files:
            module_name = file[:-3]  # Remove '.py' from filename
            if dir_path == user_scripts_dir:
                # For user scripts, adjust the module name appropriately
                module_full_path = os.path.join(dir_path, file)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_full_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except ModuleNotFoundError as e:
                    # Handle missing module imports
                    missing_package = str(e).split("'")[1]  # Extract the name of the missing package
                    print(f"Warning: skipped loading python file '{module_full_path}'!")
                    print(
                        f"'{file}' imports '{missing_package}', but '{missing_package}' is not installed locally - install python package '{missing_package}' to link functions from '{file}' to MemGPT."
                    )
                    continue
                except SyntaxError as e:
                    # Handle syntax errors in the module
                    print(f"Warning: skipped loading python file '{file}' due to a syntax error: {e}")
                    continue
                except Exception as e:
                    # Handle other general exceptions
                    print(f"Warning: skipped loading python file '{file}': {e}")
                    continue
            else:
                # For built-in scripts, use the existing method
                full_module_name = f"memgpt.functions.function_sets.{module_name}"
                try:
                    module = importlib.import_module(full_module_name)
                except Exception as e:
                    # Handle other general exceptions
                    print(f"Warning: skipped loading python module '{full_module_name}': {e}")
                    continue

            try:
                # Load the function set
                function_set = load_function_set(module)
                schemas_and_functions[module_name] = function_set
            except ValueError as e:
                print(f"Error loading function set '{module_name}': {e}")

    if merge:
        # Put all functions from all sets into the same level dict
        merged_functions = {}
        for set_name, function_set in schemas_and_functions.items():
            for function_name, function_info in function_set.items():
                if function_name in merged_functions:
                    raise ValueError(f"Duplicate function name '{function_name}' found in function set '{set_name}'")
                merged_functions[function_name] = function_info
        return merged_functions
    else:
        # Nested dict where the top level is organized by the function set name
        return schemas_and_functions
