from datetime import datetime
import json
import random
import uuid
import hashlib
import inspect
from functools import wraps
from typing import get_type_hints, Union, _GenericAlias


from urllib.parse import urlparse
import difflib
import demjson3 as demjson
import pytz
import tiktoken

from memgpt.constants import (
    JSON_LOADS_STRICT,
    FUNCTION_RETURN_CHAR_LIMIT,
    CLI_WARNING_PREFIX,
    JSON_ENSURE_ASCII,
)
from memgpt.models.chat_completion_response import ChatCompletionResponse


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
    "kindhearted",
    "lively",
    "miraculous",
    "neat",
    "openminded",
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


def get_tool_call_id() -> str:
    return str(uuid.uuid4())


def is_optional_type(hint):
    """Check if the type hint is an Optional type."""
    if isinstance(hint, _GenericAlias):
        return hint.__origin__ is Union and type(None) in hint.__args__
    return False


def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints, excluding the return type hint
        hints = {k: v for k, v in get_type_hints(func).items() if k != "return"}

        # Get the function's argument names
        arg_names = inspect.getfullargspec(func).args

        # Pair each argument with its corresponding type hint
        args_with_hints = dict(zip(arg_names[1:], args[1:]))  # Skipping 'self'

        # Check types of arguments
        for arg_name, arg_value in args_with_hints.items():
            hint = hints.get(arg_name)
            if hint and not isinstance(arg_value, hint) and not (is_optional_type(hint) and arg_value is None):
                raise ValueError(f"Argument {arg_name} does not match type {hint}")

        # Check types of keyword arguments
        for arg_name, arg_value in kwargs.items():
            hint = hints.get(arg_name)
            if hint and not isinstance(arg_value, hint) and not (is_optional_type(hint) and arg_value is None):
                raise ValueError(f"Argument {arg_name} does not match type {hint}")

        return func(*args, **kwargs)

    return wrapper


def create_random_username() -> str:
    """Generate a random username by combining an adjective and a noun."""
    adjective = random.choice(ADJECTIVE_BANK).capitalize()
    noun = random.choice(NOUN_BANK).capitalize()
    return adjective + noun


def verify_first_message_correctness(
    response: ChatCompletionResponse, require_send_message: bool = True, require_monologue: bool = False
) -> bool:
    """Can be used to enforce that the first message always uses send_message"""
    response_message = response.choices[0].message

    # First message should be a call to send_message with a non-empty content
    if (hasattr(response_message, "function_call") and response_message.function_call is not None) and (
        hasattr(response_message, "tool_calls") and response_message.tool_calls is not None
    ):
        printd(f"First message includes both function call AND tool call: {response_message}")
        return False
    elif hasattr(response_message, "function_call") and response_message.function_call is not None:
        function_call = response_message.function_call
    elif hasattr(response_message, "tool_calls") and response_message.tool_calls is not None:
        function_call = response_message.tool_calls[0].function
    else:
        printd(f"First message didn't include function call: {response_message}")
        return False

    function_name = function_call.name if function_call is not None else ""
    if require_send_message and function_name != "send_message" and function_name != "archival_memory_search":
        printd(f"First message function call wasn't send_message or archival_memory_search: {response_message}")
        return False

    if require_monologue and (not response_message.content or response_message.content is None or response_message.content == ""):
        printd(f"First message missing internal monologue: {response_message}")
        return False

    if response_message.content:
        ### Extras
        monologue = response_message.content

        def contains_special_characters(s):
            special_characters = '(){}[]"'
            return any(char in s for char in special_characters)

        if contains_special_characters(monologue):
            printd(f"First message internal monologue contained special characters: {response_message}")
            return False
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


def parse_json(string) -> dict:
    """Parse JSON string into JSON with both json and demjson"""
    result = None
    try:
        result = json.loads(string, strict=JSON_LOADS_STRICT)
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
                function_response_string = json.dumps(function_response_string, ensure_ascii=JSON_ENSURE_ASCII)
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


def create_uuid_from_string(val: str):
    """
    Generate consistent UUID from a string
    from: https://samos-it.com/posts/python-create-uuid-from-random-string-of-words.html
    """
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)
