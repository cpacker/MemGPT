import glob
import os

import yaml

from memgpt.constants import MEMGPT_DIR


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
        raise ValueError(
            f"Some functions in YAML are not part of the provided function set: {set(yaml_data['functions']) - set(function_set)} "
        )

    # If all checks pass
    return True


def load_yaml_file(file_path):
    """
    Load a YAML file and return the data.

    :param file_path: Path to the YAML file.
    :return: Data from the YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_all_presets():
    """Load all the preset configs in the examples directory"""

    ## Load the examples
    # Get the directory in which the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the path pattern
    example_path_pattern = os.path.join(script_directory, "examples", "*.yaml")
    # Listing all YAML files
    example_yaml_files = glob.glob(example_path_pattern)

    ## Load the user-provided presets
    # ~/.memgpt/presets/*.yaml
    user_presets_dir = os.path.join(MEMGPT_DIR, "presets")
    # Create directory if it doesn't exist
    if not os.path.exists(user_presets_dir):
        os.makedirs(user_presets_dir)
    # Construct the path pattern
    user_path_pattern = os.path.join(user_presets_dir, "*.yaml")
    # Listing all YAML files
    user_yaml_files = glob.glob(user_path_pattern)

    # Pull from both examplesa and user-provided
    all_yaml_files = example_yaml_files + user_yaml_files

    # Loading and creating a mapping from file name to YAML data
    all_yaml_data = {}
    for file_path in all_yaml_files:
        # Extracting the base file name without the '.yaml' extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_yaml_file(file_path)
        all_yaml_data[base_name] = data

    return all_yaml_data
