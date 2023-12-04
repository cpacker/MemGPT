from datetime import datetime
import difflib
import demjson3 as demjson
import json
import pytz
import os
import tiktoken
import memgpt
from memgpt.constants import MEMGPT_DIR

# TODO: what is this?
# DEBUG = True
DEBUG = False


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def united_diff(str1, str2):
    lines1 = str1.splitlines(True)
    lines2 = str2.splitlines(True)
    diff = difflib.unified_diff(lines1, lines2)
    return "".join(diff)


def get_local_time_military():
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone("America/Los_Angeles")
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    return formatted_time


def get_local_time_timezone(timezone="America/Los_Angeles"):
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone(timezone)
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire, including AM/PM
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return formatted_time


def get_local_time(timezone=None):
    if timezone is not None:
        return get_local_time_timezone(timezone)
    else:
        # Get the current time, which will be in the local timezone of the computer
        local_time = datetime.now()

        # You may format it as you desire, including AM/PM
        formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

        return formatted_time


def parse_json(string):
    """Parse JSON string into JSON with both json and demjson"""
    result = None
    try:
        result = json.loads(string)
        return result
    except Exception as e:
        print(f"Error parsing json with json package: {e}")

    try:
        result = demjson.decode(string)
        return result
    except demjson.JSONDecodeError as e:
        print(f"Error parsing json with demjson package: {e}")
        raise e


def list_agent_config_files(sort="last_modified"):
    """List all agent config files, ignoring dotfiles."""
    agent_dir = os.path.join(MEMGPT_DIR, "agents")
    files = os.listdir(agent_dir)

    # Remove dotfiles like .DS_Store
    files = [file for file in files if not file.startswith(".")]

    # Remove anything that's not a directory
    files = [file for file in files if os.path.isdir(os.path.join(agent_dir, file))]

    if sort is not None:
        if sort == "last_modified":
            # Sort the directories by last modified (most recent first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(agent_dir, x)), reverse=True)
        else:
            raise ValueError(f"Unrecognized sorting option {sort}")

    return files


def list_human_files():
    """List all humans files"""
    defaults_dir = os.path.join(memgpt.__path__[0], "humans", "examples")
    user_dir = os.path.join(MEMGPT_DIR, "humans")

    memgpt_defaults = os.listdir(defaults_dir)
    memgpt_defaults = [os.path.join(defaults_dir, f) for f in memgpt_defaults if f.endswith(".txt")]

    user_added = os.listdir(user_dir)
    user_added = [os.path.join(user_dir, f) for f in user_added]
    return memgpt_defaults + user_added


def list_persona_files():
    """List all personas files"""
    defaults_dir = os.path.join(memgpt.__path__[0], "personas", "examples")
    user_dir = os.path.join(MEMGPT_DIR, "personas")

    memgpt_defaults = os.listdir(defaults_dir)
    memgpt_defaults = [os.path.join(defaults_dir, f) for f in memgpt_defaults if f.endswith(".txt")]

    user_added = os.listdir(user_dir)
    user_added = [os.path.join(user_dir, f) for f in user_added]
    return memgpt_defaults + user_added


def get_human_text(name: str):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            return open(file_path, "r").read().strip()
    raise ValueError(f"Human {name} not found")


def get_persona_text(name: str):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            return open(file_path, "r").read().strip()

    raise ValueError(f"Persona {name} not found")


def get_human_text(name: str):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            return open(file_path, "r").read().strip()


def get_schema_diff(schema_a, schema_b):
    # Assuming f_schema and linked_function['json_schema'] are your JSON schemas
    f_schema_json = json.dumps(schema_a, indent=2)
    linked_function_json = json.dumps(schema_b, indent=2)

    # Compute the difference using difflib
    difference = list(difflib.ndiff(f_schema_json.splitlines(keepends=True), linked_function_json.splitlines(keepends=True)))

    # Filter out lines that don't represent changes
    difference = [line for line in difference if line.startswith("+ ") or line.startswith("- ")]

    return "".join(difference)
