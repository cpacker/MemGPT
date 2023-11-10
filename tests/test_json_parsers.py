import json

import memgpt.local_llm.json_parser as json_parser


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


def test_json_parsers():
    """Try various broken JSON and check that the parsers can fix it"""

    test_strings = [EXAMPLE_MISSING_CLOSING_BRACE, EXAMPLE_BAD_TOKEN_END, EXAMPLE_DOUBLE_JSON, EXAMPLE_HARD_LINE_FEEDS]

    for string in test_strings:
        try:
            json.loads(string)
            assert False, f"Test JSON string should have failed basic JSON parsing:\n{string}"
        except:
            print("String failed (expectedly)")
            try:
                json_parser.clean_json(string)
            except:
                f"Failed to repair test JSON string:\n{string}"
                raise
