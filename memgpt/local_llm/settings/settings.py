import json
import os

from memgpt.constants import JSON_ENSURE_ASCII, MEMGPT_DIR
from memgpt.local_llm.settings.deterministic_mirostat import (
    settings as det_miro_settings,
)
from memgpt.local_llm.settings.simple import settings as simple_settings

DEFAULT = "simple"
SETTINGS_FOLDER_NAME = "settings"
COMPLETION_SETTINGS_FILE_NAME = "completions_api_settings.json"


def get_completions_settings(defaults="simple") -> dict:
    """Pull from the home directory settings if they exist, otherwise default"""
    from memgpt.utils import printd

    # Load up some default base settings
    printd(f"Loading default settings from '{defaults}'")
    if defaults == "simple":
        # simple = basic stop strings
        settings = simple_settings
    elif defaults == "deterministic_mirostat":
        settings = det_miro_settings
    elif defaults is None:
        settings = dict()
    else:
        raise ValueError(defaults)

    # Check if settings_dir folder exists (if not, create it)
    settings_dir = os.path.join(MEMGPT_DIR, SETTINGS_FOLDER_NAME)
    if not os.path.exists(settings_dir):
        printd(f"Settings folder '{settings_dir}' doesn't exist, creating it...")
        try:
            os.makedirs(settings_dir)
        except Exception as e:
            print(f"Error: failed to create settings folder '{settings_dir}'.\n{e}")
            return settings

    # Then, check if settings_dir/completions_api_settings.json file exists
    settings_file = os.path.join(settings_dir, COMPLETION_SETTINGS_FILE_NAME)

    if os.path.isfile(settings_file):
        # Load into a dict called "settings"
        printd(f"Found completion settings file '{settings_file}', loading it...")
        try:
            with open(settings_file, "r", encoding="utf-8") as file:
                user_settings = json.load(file)
            if len(user_settings) > 0:
                printd(
                    f"Updating base settings with the following user settings:\n{json.dumps(user_settings,indent=2, ensure_ascii=JSON_ENSURE_ASCII)}"
                )
                settings.update(user_settings)
            else:
                printd(f"'{settings_file}' was empty, ignoring...")
        except json.JSONDecodeError as e:
            print(f"Error: failed to load user settings file '{settings_file}', invalid json.\n{e}")
        except Exception as e:
            print(f"Error: failed to load user settings file.\n{e}")

    else:
        printd(f"No completion settings file '{settings_file}', skipping...")
        # Create the file settings_file to make it easy for the user to edit
        try:
            with open(settings_file, "w", encoding="utf-8") as file:
                # We don't want to dump existing default settings in case we modify
                # the default settings in the future
                # json.dump(settings, file, indent=4)
                json.dump({}, file, indent=4)
        except Exception as e:
            print(f"Error: failed to create empty settings file '{settings_file}'.\n{e}")

    return settings
