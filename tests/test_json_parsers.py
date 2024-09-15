import letta.local_llm.json_parser as json_parser
from letta.utils import json_loads

EXAMPLE_ESCAPED_UNDERSCORES = """{
  "function":"send\_message",
  "params": {
    "inner\_thoughts": "User is asking for information about themselves. Retrieving data from core memory.",
    "message": "I know that you are Chad. Is there something specific you would like to know or talk about regarding yourself?"
"""


EXAMPLE_MISSING_CLOSING_BRACE = """{
  "function": "send_message",
  "params": {
    "inner_thoughts": "Oops, I got their name wrong! I should apologize and correct myself.",
    "message": "Sorry about that! I assumed you were Chad. Welcome, Brad! "
  }
"""

EXAMPLE_BAD_TOKEN_END = """{
  "function": "send_message",
  "params": {
    "inner_thoughts": "Oops, I got their name wrong! I should apologize and correct myself.",
    "message": "Sorry about that! I assumed you were Chad. Welcome, Brad! "
  }
}<|>"""

EXAMPLE_DOUBLE_JSON = """{
  "function": "core_memory_append",
  "params": {
    "name": "human",
    "content": "Brad, 42 years old, from Germany."
  }
}
{
  "function": "send_message",
  "params": {
    "message": "Got it! Your age and nationality are now saved in my memory."
  }
}
"""

EXAMPLE_HARD_LINE_FEEDS = """{
  "function": "send_message",
  "params": {
    "message": "Let's create a list:
- First, we can do X
- Then, we can do Y!
- Lastly, we can do Z :)"
  }
}
"""

# Situation where beginning of send_message call is fine (and thus can be extracted)
# but has a long training garbage string that comes after
EXAMPLE_SEND_MESSAGE_PREFIX_OK_REST_BAD = """{
  "function": "send_message",
  "params": {
    "inner_thoughts": "User request for debug assistance",
    "message": "Of course, Chad. Please check the system log file for 'assistant.json' and send me the JSON output you're getting. Armed with that data, I'll assist you in debugging the issue.",
GARBAGEGARBAGEGARBAGEGARBAGE
GARBAGEGARBAGEGARBAGEGARBAGE
GARBAGEGARBAGEGARBAGEGARBAGE
"""

EXAMPLE_ARCHIVAL_SEARCH = """

{
  "function": "archival_memory_search",
  "params": {
    "inner_thoughts": "Looking for WaitingForAction.",
    "query": "WaitingForAction",
"""


def test_json_parsers():
    """Try various broken JSON and check that the parsers can fix it"""

    test_strings = [
        EXAMPLE_ESCAPED_UNDERSCORES,
        EXAMPLE_MISSING_CLOSING_BRACE,
        EXAMPLE_BAD_TOKEN_END,
        EXAMPLE_DOUBLE_JSON,
        EXAMPLE_HARD_LINE_FEEDS,
        EXAMPLE_SEND_MESSAGE_PREFIX_OK_REST_BAD,
        EXAMPLE_ARCHIVAL_SEARCH,
    ]

    for string in test_strings:
        try:
            json_loads(string)
            assert False, f"Test JSON string should have failed basic JSON parsing:\n{string}"
        except:
            print("String failed (expectedly)")
            try:
                json_parser.clean_json(string)
            except:
                f"Failed to repair test JSON string:\n{string}"
                raise
