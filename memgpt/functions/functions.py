import importlib
import inspect
import os


from memgpt.functions.schema_generator import generate_schema
from memgpt.constants import MEMGPT_DIR


def load_function_set(set_name):
    """Load the functions and generate schema for them"""
    function_dict = {}

    module_name = f"memgpt.functions.function_sets.{set_name}"
    base_functions = importlib.import_module(module_name)

    for attr_name in dir(base_functions):
        # Get the attribute
        attr = getattr(base_functions, attr_name)

        # Check if it's a callable function and not a built-in or special method
        if inspect.isfunction(attr) and attr.__module__ == base_functions.__name__:
            if attr_name in function_dict:
                raise ValueError(f"Found a duplicate of function name '{attr_name}'")

            generated_schema = generate_schema(attr)
            function_dict[attr_name] = {
                "python_function": attr,
                "json_schema": generated_schema,
            }

    if len(function_dict) == 0:
        raise ValueError(f"No functions found in module {module_name}")
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
    all_module_files = example_module_files + user_module_files

    schemas_and_functions = {}
    for file in all_module_files:
        # Convert filename to module name
        module_name = f"memgpt.functions.function_sets.{file[:-3]}"  # Remove '.py' from filename

        try:
            # Load the function set
            function_set = load_function_set(file[:-3])  # Pass the module part of the name
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
