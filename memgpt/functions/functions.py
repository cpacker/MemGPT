import importlib
import inspect


from memgpt.config import MemGPTConfig
from memgpt.functions.schema_generator import generate_schema
from memgpt.constants import CLI_WARNING_PREFIX


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


def load_all_function_sets():
    schemas_and_functions = {}
    for full_module_name in MemGPTConfig.load().functions_modules:
        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            # Handle other general exceptions
            print(f"{CLI_WARNING_PREFIX}skipped loading python module '{full_module_name}': {e}")
            continue

        try:
            # Load the function set
            function_set = load_function_set(module)
            for function_name, function_info in function_set.items():
                if function_name in schemas_and_functions:
                    raise ValueError(f"Duplicate function name '{function_name}' found in function set '{full_module_name}'")
                schemas_and_functions[function_name] = function_info

        except ValueError as e:
            print(f"Error loading function set, skipping '{full_module_name}': {e}")
    return schemas_and_functions
