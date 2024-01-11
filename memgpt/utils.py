from datetime import datetime
import re
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import io
from urllib.parse import urlparse
from contextlib import contextmanager
import difflib
import demjson3 as demjson
import pytz
import tiktoken

import memgpt
from memgpt.constants import (
    MEMGPT_DIR,
    FUNCTION_RETURN_CHAR_LIMIT,
    CLI_WARNING_PREFIX,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
)

from memgpt.openai_backcompat.openai_object import OpenAIObject

# TODO: what is this?
# DEBUG = True
DEBUG = False

ADJECTIVE_BANK = [
    "beautiful",
    "gentle",
    "angry",
    "vivacious",
    "grumpy",
    "luxurious",
    "fierce",
    "delicate",
    "fluffy",
    "radiant",
    "elated",
    "magnificent",
    "sassy",
    "ecstatic",
    "lustrous",
    "gleaming",
    "sorrowful",
    "majestic",
    "proud",
    "dynamic",
    "energetic",
    "mysterious",
    "loyal",
    "brave",
    "decisive",
    "frosty",
    "cheerful",
    "adorable",
    "melancholy",
    "vibrant",
    "elegant",
    "gracious",
    "inquisitive",
    "opulent",
    "peaceful",
    "rebellious",
    "scintillating",
    "dazzling",
    "whimsical",
    "impeccable",
    "meticulous",
    "resilient",
    "charming",
    "vivacious",
    "creative",
    "intuitive",
    "compassionate",
    "innovative",
    "enthusiastic",
    "tremendous",
    "effervescent",
    "tenacious",
    "fearless",
    "sophisticated",
    "witty",
    "optimistic",
    "exquisite",
    "sincere",
    "generous",
    "kindhearted",
    "serene",
    "amiable",
    "adventurous",
    "bountiful",
    "courageous",
    "diligent",
    "exotic",
    "grateful",
    "harmonious",
    "imaginative",
    "jubilant",
    "keen",
    "luminous",
    "nurturing",
    "outgoing",
    "passionate",
    "quaint",
    "resourceful",
    "sturdy",
    "tactful",
    "unassuming",
    "versatile",
    "wondrous",
    "youthful",
    "zealous",
    "ardent",
    "benevolent",
    "capricious",
    "dedicated",
    "empathetic",
    "fabulous",
    "gregarious",
    "humble",
    "intriguing",
    "jovial",
    "kind",
    "lovable",
    "mindful",
    "noble",
    "original",
    "pleasant",
    "quixotic",
    "reliable",
    "spirited",
    "tranquil",
    "unique",
    "venerable",
    "warmhearted",
    "xenodochial",
    "yearning",
    "zesty",
    "amusing",
    "blissful",
    "calm",
    "daring",
    "enthusiastic",
    "faithful",
    "graceful",
    "honest",
    "incredible",
    "joyful",
    "kind",
    "lovely",
    "merry",
    "noble",
    "optimistic",
    "peaceful",
    "quirky",
    "respectful",
    "sweet",
    "trustworthy",
    "understanding",
    "vibrant",
    "witty",
    "xenial",
    "youthful",
    "zealous",
    "ambitious",
    "brilliant",
    "careful",
    "devoted",
    "energetic",
    "friendly",
    "glorious",
    "humorous",
    "intelligent",
    "jovial",
    "knowledgeable",
    "loyal",
    "modest",
    "nice",
    "obedient",
    "patient",
    "quiet",
    "resilient",
    "selfless",
    "tolerant",
    "unique",
    "versatile",
    "warm",
    "xerothermic",
    "yielding",
    "zestful",
    "amazing",
    "bold",
    "charming",
    "determined",
    "exciting",
    "funny",
    "happy",
    "imaginative",
    "jolly",
    "keen",
    "loving",
    "magnificent",
    "nifty",
    "outstanding",
    "polite",
    "quick",
    "reliable",
    "sincere",
    "thoughtful",
    "unusual",
    "valuable",
    "wonderful",
    "xenodochial",
    "zealful",
    "admirable",
    "bright",
    "clever",
    "dedicated",
    "extraordinary",
    "generous",
    "hardworking",
    "inspiring",
    "jubilant",
    "kind-hearted",
    "lively",
    "miraculous",
    "neat",
    "open-minded",
    "passionate",
    "remarkable",
    "stunning",
    "truthful",
    "upbeat",
    "vivacious",
    "welcoming",
    "yare",
    "zealous",
]

NOUN_BANK = [
    "lizard",
    "firefighter",
    "banana",
    "castle",
    "dolphin",
    "elephant",
    "forest",
    "giraffe",
    "harbor",
    "iceberg",
    "jewelry",
    "kangaroo",
    "library",
    "mountain",
    "notebook",
    "orchard",
    "penguin",
    "quilt",
    "rainbow",
    "squirrel",
    "teapot",
    "umbrella",
    "volcano",
    "waterfall",
    "xylophone",
    "yacht",
    "zebra",
    "apple",
    "butterfly",
    "caterpillar",
    "dragonfly",
    "elephant",
    "flamingo",
    "gorilla",
    "hippopotamus",
    "iguana",
    "jellyfish",
    "koala",
    "lemur",
    "mongoose",
    "nighthawk",
    "octopus",
    "panda",
    "quokka",
    "rhinoceros",
    "salamander",
    "tortoise",
    "unicorn",
    "vulture",
    "walrus",
    "xenopus",
    "yak",
    "zebu",
    "asteroid",
    "balloon",
    "compass",
    "dinosaur",
    "eagle",
    "firefly",
    "galaxy",
    "hedgehog",
    "island",
    "jaguar",
    "kettle",
    "lion",
    "mammoth",
    "nucleus",
    "owl",
    "pumpkin",
    "quasar",
    "reindeer",
    "snail",
    "tiger",
    "universe",
    "vampire",
    "wombat",
    "xerus",
    "yellowhammer",
    "zeppelin",
    "alligator",
    "buffalo",
    "cactus",
    "donkey",
    "emerald",
    "falcon",
    "gazelle",
    "hamster",
    "icicle",
    "jackal",
    "kitten",
    "leopard",
    "mushroom",
    "narwhal",
    "opossum",
    "peacock",
    "quail",
    "rabbit",
    "scorpion",
    "toucan",
    "urchin",
    "viper",
    "wolf",
    "xray",
    "yucca",
    "zebu",
    "acorn",
    "biscuit",
    "cupcake",
    "daisy",
    "eyeglasses",
    "frisbee",
    "goblin",
    "hamburger",
    "icicle",
    "jackfruit",
    "kaleidoscope",
    "lighthouse",
    "marshmallow",
    "nectarine",
    "obelisk",
    "pancake",
    "quicksand",
    "raspberry",
    "spinach",
    "truffle",
    "umbrella",
    "volleyball",
    "walnut",
    "xylophonist",
    "yogurt",
    "zucchini",
    "asterisk",
    "blackberry",
    "chimpanzee",
    "dumpling",
    "espresso",
    "fireplace",
    "gnome",
    "hedgehog",
    "illustration",
    "jackhammer",
    "kumquat",
    "lemongrass",
    "mandolin",
    "nugget",
    "ostrich",
    "parakeet",
    "quiche",
    "racquet",
    "seashell",
    "tadpole",
    "unicorn",
    "vaccination",
    "wolverine",
    "xenophobia",
    "yam",
    "zeppelin",
    "accordion",
    "broccoli",
    "carousel",
    "daffodil",
    "eggplant",
    "flamingo",
    "grapefruit",
    "harpsichord",
    "impression",
    "jackrabbit",
    "kitten",
    "llama",
    "mandarin",
    "nachos",
    "obelisk",
    "papaya",
    "quokka",
    "rooster",
    "sunflower",
    "turnip",
    "ukulele",
    "viper",
    "waffle",
    "xylograph",
    "yeti",
    "zephyr",
    "abacus",
    "blueberry",
    "crocodile",
    "dandelion",
    "echidna",
    "fig",
    "giraffe",
    "hamster",
    "iguana",
    "jackal",
    "kiwi",
    "lobster",
    "marmot",
    "noodle",
    "octopus",
    "platypus",
    "quail",
    "raccoon",
    "starfish",
    "tulip",
    "urchin",
    "vampire",
    "walrus",
    "xylophone",
    "yak",
    "zebra",
]


def create_random_username() -> str:
    """Generate a random username by combining an adjective and a noun."""
    adjective = random.choice(ADJECTIVE_BANK).capitalize()
    noun = random.choice(NOUN_BANK).capitalize()
    return adjective + noun


def verify_first_message_correctness(response, require_send_message=True, require_monologue=False) -> bool:
    """Can be used to enforce that the first message always uses send_message"""
    response_message = response.choices[0].message

    # First message should be a call to send_message with a non-empty content
    if require_send_message and not response_message.get("function_call"):
        printd(f"First message didn't include function call: {response_message}")
        return False

    function_call = response_message.get("function_call")
    function_name = function_call.get("name") if function_call is not None else ""
    if require_send_message and function_name != "send_message" and function_name != "archival_memory_search":
        printd(f"First message function call wasn't send_message or archival_memory_search: {response_message}")
        return False

    if require_monologue and (
        not response_message.get("content") or response_message["content"] is None or response_message["content"] == ""
    ):
        printd(f"First message missing internal monologue: {response_message}")
        return False

    if response_message.get("content"):
        ### Extras
        monologue = response_message.get("content")

        def contains_special_characters(s):
            special_characters = '(){}[]"'
            return any(char in s for char in special_characters)

        if contains_special_characters(monologue):
            printd(f"First message internal monologue contained special characters: {response_message}")
            return False
        # if 'functions' in monologue or 'send_message' in monologue or 'inner thought' in monologue.lower():
        if "functions" in monologue or "send_message" in monologue:
            # Sometimes the syntax won't be correct and internal syntax will leak into message.context
            printd(f"First message internal monologue contained reserved words: {response_message}")
            return False

    return True


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@contextmanager
def suppress_stdout():
    """Used to temporarily stop stdout (eg for the 'MockLLM' message)"""
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout


def open_folder_in_explorer(folder_path):
    """
    Opens the specified folder in the system's native file explorer.

    :param folder_path: Absolute path to the folder to be opened.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The specified folder {folder_path} does not exist.")

    # Determine the operating system
    os_name = platform.system()

    # Open the folder based on the operating system
    if os_name == "Windows":
        # Windows: use 'explorer' command
        subprocess.run(["explorer", folder_path], check=True)
    elif os_name == "Darwin":
        # macOS: use 'open' command
        subprocess.run(["open", folder_path], check=True)
    elif os_name == "Linux":
        # Linux: use 'xdg-open' command (works for most Linux distributions)
        subprocess.run(["xdg-open", folder_path], check=True)
    else:
        raise OSError(f"Unsupported operating system {os_name}.")


# Custom unpickler
class OpenAIBackcompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "openai.openai_object":
            return OpenAIObject
        return super().find_class(module, name)


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


def parse_formatted_time(formatted_time):
    # parse times returned by memgpt.utils.get_formatted_time()
    return datetime.strptime(formatted_time, "%Y-%m-%d %I:%M:%S %p %Z%z")


def datetime_to_timestamp(dt):
    # convert datetime object to integer timestamp
    return int(dt.timestamp())


def timestamp_to_datetime(ts):
    # convert integer timestamp to datetime object
    return datetime.fromtimestamp(ts)


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
        time_str = get_local_time_timezone(timezone)
    else:
        # Get the current time, which will be in the local timezone of the computer
        local_time = datetime.now().astimezone()

        # You may format it as you desire, including AM/PM
        time_str = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return time_str.strip()


def format_datetime(dt):
    return dt.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")


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


def validate_function_response(function_response_string: any, strict: bool = False, truncate: bool = True) -> str:
    """Check to make sure that a function used by MemGPT returned a valid response

    Responses need to be strings (or None) that fall under a certain text count limit.
    """
    if not isinstance(function_response_string, str):
        # Soft correction for a few basic types

        if function_response_string is None:
            # function_response_string = "Empty (no function output)"
            function_response_string = "None"  # backcompat

        elif isinstance(function_response_string, dict):
            if strict:
                # TODO add better error message
                raise ValueError(function_response_string)

            # Allow dict through since it will be cast to json.dumps()
            try:
                # TODO find a better way to do this that won't result in double escapes
                function_response_string = json.dumps(function_response_string)
            except:
                raise ValueError(function_response_string)

        else:
            if strict:
                # TODO add better error message
                raise ValueError(function_response_string)

            # Try to convert to a string, but throw a warning to alert the user
            try:
                function_response_string = str(function_response_string)
            except:
                raise ValueError(function_response_string)

    # Now check the length and make sure it doesn't go over the limit
    # TODO we should change this to a max token limit that's variable based on tokens remaining (or context-window)
    if truncate and len(function_response_string) > FUNCTION_RETURN_CHAR_LIMIT:
        print(
            f"{CLI_WARNING_PREFIX}function return was over limit ({len(function_response_string)} > {FUNCTION_RETURN_CHAR_LIMIT}) and was truncated"
        )
        function_response_string = f"{function_response_string[:FUNCTION_RETURN_CHAR_LIMIT]}... [NOTE: function output was truncated since it exceeded the character limit ({len(function_response_string)} > {FUNCTION_RETURN_CHAR_LIMIT})]"

    return function_response_string


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


def get_human_text(name: str, enforce_limit=True):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            human_text = open(file_path, "r").read().strip()
            if enforce_limit and len(human_text) > CORE_MEMORY_HUMAN_CHAR_LIMIT:
                raise ValueError(f"Contents of {name}.txt is over the character limit ({len(human_text)} > {CORE_MEMORY_HUMAN_CHAR_LIMIT})")
            return human_text

    raise ValueError(f"Human {name}.txt not found")


def get_persona_text(name: str, enforce_limit=True):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            persona_text = open(file_path, "r").read().strip()
            if enforce_limit and len(persona_text) > CORE_MEMORY_PERSONA_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(persona_text)} > {CORE_MEMORY_PERSONA_CHAR_LIMIT})"
                )
            return persona_text

    raise ValueError(f"Persona {name}.txt not found")


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


# datetime related
def validate_date_format(date_str):
    """Validate the given date string in the format 'YYYY-MM-DD'."""
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def extract_date_from_timestamp(timestamp):
    """Extracts and returns the date from the given timestamp."""
    # Extracts the date (ignoring the time and timezone)
    match = re.match(r"(\d{4}-\d{2}-\d{2})", timestamp)
    return match.group(1) if match else None
