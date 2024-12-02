import os
import warnings
from typing import List, Union

import requests
import tiktoken

import letta.local_llm.llm_chat_completion_wrappers.airoboros as airoboros
import letta.local_llm.llm_chat_completion_wrappers.chatml as chatml
import letta.local_llm.llm_chat_completion_wrappers.configurable_wrapper as configurable_wrapper
import letta.local_llm.llm_chat_completion_wrappers.dolphin as dolphin
import letta.local_llm.llm_chat_completion_wrappers.llama3 as llama3
import letta.local_llm.llm_chat_completion_wrappers.zephyr as zephyr
from letta.schemas.openai.chat_completion_request import Tool, ToolCall


def post_json_auth_request(uri, json_payload, auth_type, auth_key):
    """Send a POST request with a JSON payload and optional authentication"""

    # By default most local LLM inference servers do not have authorization enabled
    if auth_type is None or auth_type == "":
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


# TODO: support tokenizers/tokenizer apis available in local models
def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def num_tokens_from_functions(functions: List[dict], model: str = "gpt-4"):
    """Return the number of tokens used by a list of functions.

    Copied from https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        from letta.utils import printd

        printd(f"Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for function in functions:
        function_tokens = len(encoding.encode(function["name"]))
        if function["description"]:
            if not isinstance(function["description"], str):
                warnings.warn(f"Function {function['name']} has non-string description: {function['description']}")
            else:
                function_tokens += len(encoding.encode(function["description"]))
        else:
            warnings.warn(f"Function {function['name']} has no description, function: {function}")

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(o))
                        else:
                            warnings.warn(f"num_tokens_from_functions: Unsupported field {field} in function {function}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens


def num_tokens_from_tool_calls(tool_calls: Union[List[dict], List[ToolCall]], model: str = "gpt-4"):
    """Based on above code (num_tokens_from_functions).

    Example to encode:
    [{
        'id': '8b6707cf-2352-4804-93db-0423f',
        'type': 'function',
        'function': {
            'name': 'send_message',
            'arguments': '{\n  "message": "More human than human is our motto."\n}'
        }
    }]
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_call_id = tool_call["id"]
            tool_call_type = tool_call["type"]
            tool_call_function = tool_call["function"]
            tool_call_function_name = tool_call_function["name"]
            tool_call_function_arguments = tool_call_function["arguments"]
        elif isinstance(tool_call, Tool):
            tool_call_id = tool_call.id
            tool_call_type = tool_call.type
            tool_call_function = tool_call.function
            tool_call_function_name = tool_call_function.name
            tool_call_function_arguments = tool_call_function.arguments
        else:
            raise ValueError(f"Unknown tool call type: {type(tool_call)}")

        function_tokens = len(encoding.encode(tool_call_id))
        function_tokens += 2 + len(encoding.encode(tool_call_type))
        function_tokens += 2 + len(encoding.encode(tool_call_function_name))
        function_tokens += 2 + len(encoding.encode(tool_call_function_arguments))

        num_tokens += function_tokens

    # TODO adjust?
    num_tokens += 12
    return num_tokens


def num_tokens_from_messages(messages: List[dict], model: str = "gpt-4") -> int:
    """Return the number of tokens used by a list of messages.

    From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For counting tokens in function calling RESPONSES, see:
        https://hmarr.com/blog/counting-openai-tokens/, https://github.com/hmarr/openai-chat-tokens

    For counting tokens in function calling REQUESTS, see:
        https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
    """
    try:
        # Attempt to search for the encoding based on the model string
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        from letta.utils import printd

        printd(
            f"num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
        # raise NotImplementedError(
        # f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        # )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            try:

                if isinstance(value, list) and key == "tool_calls":
                    num_tokens += num_tokens_from_tool_calls(tool_calls=value, model=model)
                    # special case for tool calling (list)
                    # num_tokens += len(encoding.encode(value["name"]))
                    # num_tokens += len(encoding.encode(value["arguments"]))

                else:
                    if value is None:
                        # raise ValueError(f"Message has null value: {key} with value: {value} - message={message}")
                        warnings.warn(f"Message has null value: {key} with value: {value} - message={message}")
                    else:
                        if not isinstance(value, str):
                            raise ValueError(f"Message has non-string value: {key} with value: {value} - message={message}")
                        num_tokens += len(encoding.encode(value))

                if key == "name":
                    num_tokens += tokens_per_name

            except TypeError as e:
                print(f"tiktoken encoding failed on: {value}")
                raise e

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_available_wrappers() -> dict:
    return {
        "llama3": llama3.LLaMA3InnerMonologueWrapper(),
        "llama3-grammar": llama3.LLaMA3InnerMonologueWrapper(),
        "llama3-hints-grammar": llama3.LLaMA3InnerMonologueWrapper(assistant_prefix_hint=True),
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
