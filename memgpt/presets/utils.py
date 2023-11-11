import os
import glob
import yaml


# def is_valid_yaml_format(yaml_data, function_set):
#     """
#     Check if the given YAML data follows the specified format and if all functions in the yaml are part of the function_set.

#     :param yaml_data: The data loaded from a YAML file.
#     :param function_set: A set of valid function names.
#     :return: True if valid, False otherwise.
#     """
#     print("function_set", function_set)
#     # Check for required keys
#     if not all(key in yaml_data for key in ["system_prompt", "functions"]):
#         return False

#     # Check if 'functions' is a list of strings
#     if not all(isinstance(item, str) for item in yaml_data.get("functions", [])):
#         return False

#     # Check if all functions in YAML are part of function_set
#     if not set(yaml_data["functions"]).issubset(function_set):
#         return False

#     return True


def is_valid_yaml_format(yaml_data, function_set):
    """
    Check if the given YAML data follows the specified format and if all functions in the yaml are part of the function_set.
    Raises ValueError if any check fails.

    :param yaml_data: The data loaded from a YAML file.
    :param function_set: A set of valid function names.
    """
    # Check for required keys
    if not all(key in yaml_data for key in ["system_prompt", "functions"]):
        raise ValueError("YAML data is missing one or more required keys: 'system_prompt', 'functions'.")

    # Check if 'functions' is a list of strings
    if not all(isinstance(item, str) for item in yaml_data.get("functions", [])):
        raise ValueError("'functions' should be a list of strings.")

    # Check if all functions in YAML are part of function_set
    if not set(yaml_data["functions"]).issubset(function_set):
        raise ValueError("Some functions in YAML are not part of the provided function set.")

    # If all checks pass
    return True


def load_yaml_file(file_path):
    """
    Load a YAML file and return the data.

    :param file_path: Path to the YAML file.
    :return: Data from the YAML file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_all_examples():
    """Load all the preset configs in the examples directory"""
    # Get the directory in which the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the path pattern
    path_pattern = os.path.join(script_directory, "examples", "*.yaml")

    # Listing all YAML files
    yaml_files = glob.glob(path_pattern)

    # Loading and creating a mapping from file name to YAML data
    all_yaml_data = {}
    for file_path in yaml_files:
        # Extracting the base file name without the '.yaml' extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_yaml_file(file_path)
        all_yaml_data[base_name] = data

    return all_yaml_data
