import os
import tiktoken

import memgpt.local_llm.llm_chat_completion_wrappers.airoboros as airoboros
import memgpt.local_llm.llm_chat_completion_wrappers.dolphin as dolphin
import memgpt.local_llm.llm_chat_completion_wrappers.zephyr as zephyr


class DotDict(dict):
    """Allow dot access on properties similar to OpenAI response object"""

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value

    # following methods necessary for pickling
    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


def load_grammar_file(grammar):
    # Set grammar
    grammar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", f"{grammar}.gbnf")

    # Check if the file exists
    if not os.path.isfile(grammar_file):
        # If the file doesn't exist, raise a FileNotFoundError
        raise FileNotFoundError(f"The grammar file {grammar_file} does not exist.")

    with open(grammar_file, "r") as file:
        grammar_str = file.read()

    return grammar_str


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def get_available_wrappers() -> dict:
    return {
        "airoboros-l2-70b-2.1": airoboros.Airoboros21InnerMonologueWrapper(),
        "airoboros-l2-70b-2.1-grammar": airoboros.Airoboros21InnerMonologueWrapper(include_opening_brace_in_prefix=False),
        "dolphin-2.1-mistral-7b": dolphin.Dolphin21MistralWrapper(),
        "dolphin-2.1-mistral-7b-grammar": dolphin.Dolphin21MistralWrapper(include_opening_brace_in_prefix=False),
        "zephyr-7B": zephyr.ZephyrMistralInnerMonologueWrapper(),
        "zephyr-7B-grammar": zephyr.ZephyrMistralInnerMonologueWrapper(include_opening_brace_in_prefix=False),
    }
