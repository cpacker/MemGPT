import os
import requests
import tiktoken

import memgpt.local_llm.llm_chat_completion_wrappers.airoboros as airoboros
import memgpt.local_llm.llm_chat_completion_wrappers.dolphin as dolphin
import memgpt.local_llm.llm_chat_completion_wrappers.zephyr as zephyr
import memgpt.local_llm.llm_chat_completion_wrappers.chatml as chatml
import memgpt.local_llm.llm_chat_completion_wrappers.configurable_wrapper as configurable_wrapper


def post_json_auth_request(uri, json_payload, auth_type, auth_key):
    """Send a POST request with a JSON payload and optional authentication"""

    # By default most local LLM inference servers do not have authorization enabled
    if auth_type is None:
        response = requests.post(uri, json=json_payload)

    # Used by OpenAI, together.ai, Mistral AI
    elif auth_type == "bearer_token":
        if auth_key is None:
            raise ValueError(f"auth_type is {auth_type}, but auth_key is null")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {auth_key}"}
        response = requests.post(uri, json=json_payload, headers=headers)

    # Used by OpenAI Azure
    elif auth_type == "api_key":
        if auth_key is None:
            raise ValueError(f"auth_type is {auth_type}, but auth_key is null")
        headers = {"Content-Type": "application/json", "api-key": f"{auth_key}"}
        response = requests.post(uri, json=json_payload, headers=headers)

    else:
        raise ValueError(f"Unsupport authentication type: {auth_type}")

    return response


# deprecated for Box
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

    with open(grammar_file, "r", encoding="utf-8") as file:
        grammar_str = file.read()

    return grammar_str


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def get_available_wrappers() -> dict:
    return {
        "experimental-wrapper-neural-chat-grammar-noforce": configurable_wrapper.ConfigurableJSONWrapper(
            post_prompt="### Assistant:",
            sys_prompt_start="### System:\n",
            sys_prompt_end="\n",
            user_prompt_start="### User:\n",
            user_prompt_end="\n",
            assistant_prompt_start="### Assistant:\n",
            assistant_prompt_end="\n",
            tool_prompt_start="### User:\n",
            tool_prompt_end="\n",
            strip_prompt=True,
        ),
        # New chatml-based wrappers
        "chatml": chatml.ChatMLInnerMonologueWrapper(),
        "chatml-grammar": chatml.ChatMLInnerMonologueWrapper(),
        "chatml-noforce": chatml.ChatMLOuterInnerMonologueWrapper(),
        "chatml-noforce-grammar": chatml.ChatMLOuterInnerMonologueWrapper(),
        # "chatml-noforce-sysm": chatml.ChatMLOuterInnerMonologueWrapper(use_system_role_in_user=True),
        "chatml-noforce-roles": chatml.ChatMLOuterInnerMonologueWrapper(use_system_role_in_user=True, allow_function_role=True),
        "chatml-noforce-roles-grammar": chatml.ChatMLOuterInnerMonologueWrapper(use_system_role_in_user=True, allow_function_role=True),
        # With extra hints
        "chatml-hints": chatml.ChatMLInnerMonologueWrapper(assistant_prefix_hint=True),
        "chatml-hints-grammar": chatml.ChatMLInnerMonologueWrapper(assistant_prefix_hint=True),
        "chatml-noforce-hints": chatml.ChatMLOuterInnerMonologueWrapper(assistant_prefix_hint=True),
        "chatml-noforce-hints-grammar": chatml.ChatMLOuterInnerMonologueWrapper(assistant_prefix_hint=True),
        # Legacy wrappers
        "airoboros-l2-70b-2.1": airoboros.Airoboros21InnerMonologueWrapper(),
        "airoboros-l2-70b-2.1-grammar": airoboros.Airoboros21InnerMonologueWrapper(assistant_prefix_extra=None),
        "dolphin-2.1-mistral-7b": dolphin.Dolphin21MistralWrapper(),
        "dolphin-2.1-mistral-7b-grammar": dolphin.Dolphin21MistralWrapper(include_opening_brace_in_prefix=False),
        "zephyr-7B": zephyr.ZephyrMistralInnerMonologueWrapper(),
        "zephyr-7B-grammar": zephyr.ZephyrMistralInnerMonologueWrapper(include_opening_brace_in_prefix=False),
    }
