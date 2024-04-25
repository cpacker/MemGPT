from typing import List
from memgpt.utils import printd
from memgpt.functions.functions import load_all_function_sets


def generate_functions_json(preset_functions: List[str]):
    """
    Generate JSON schema for the functions based on what is locally available.

    TODO: store function definitions in the DB, instead of locally
    """
    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    available_functions = load_all_function_sets()
    # Filter down the function set based on what the preset requested
    preset_function_set = {}
    for f_name in preset_functions:
        if f_name not in available_functions:
            raise ValueError(f"Function '{f_name}' was specified in preset, but is not in function library:\n{available_functions.keys()}")
        preset_function_set[f_name] = available_functions[f_name]
    assert len(preset_functions) == len(preset_function_set)
    preset_function_set_schemas = [f_dict["json_schema"] for f_name, f_dict in preset_function_set.items()]
    printd(f"Available functions:\n", list(preset_function_set.keys()))
    return preset_function_set_schemas
