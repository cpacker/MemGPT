"""Key idea: create drop-in replacement for agent's ChatCompletion call that runs on an OpenLLM backend"""

import os
import requests
import json

from .webui.api import get_webui_completion
from .lmstudio.api import get_lmstudio_completion
from .llamacpp.api import get_llamacpp_completion
from .koboldcpp.api import get_koboldcpp_completion
from .ollama.api import get_ollama_completion
from .llm_chat_completion_wrappers import airoboros, dolphin, zephyr, simple_summary_wrapper
from .utils import DotDict
from ..prompts.gpt_summarize import SYSTEM as SUMMARIZE_SYSTEM_MESSAGE
from ..errors import LocalLLMConnectionError, LocalLLMError

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
DEBUG = False
# DEBUG = True
DEFAULT_WRAPPER = airoboros.Airoboros21InnerMonologueWrapper
has_shown_warning = False


def get_chat_completion(
    model,  # no model, since the model is fixed to whatever you set in your own backend
    messages,
    functions=None,
    function_call="auto",
    context_window=None,
):
    assert context_window is not None, "Local LLM calls need the context length to be explicitly set"
    global has_shown_warning
    grammar_name = None

    if HOST is None:
        raise ValueError(f"The OPENAI_API_BASE environment variable is not defined. Please set it in your environment.")
    if HOST_TYPE is None:
        raise ValueError(f"The BACKEND_TYPE environment variable is not defined. Please set it in your environment.")

    if function_call != "auto":
        raise ValueError(f"function_call == {function_call} not supported (auto only)")

    if messages[0]["role"] == "system" and messages[0]["content"].strip() == SUMMARIZE_SYSTEM_MESSAGE.strip():
        # Special case for if the call we're making is coming from the summarizer
        llm_wrapper = simple_summary_wrapper.SimpleSummaryWrapper()
    elif model == "airoboros-l2-70b-2.1":
        llm_wrapper = airoboros.Airoboros21InnerMonologueWrapper()
    elif model == "airoboros-l2-70b-2.1-grammar":
        llm_wrapper = airoboros.Airoboros21InnerMonologueWrapper(include_opening_brace_in_prefix=False)
        # grammar_name = "json"
        grammar_name = "json_func_calls_with_inner_thoughts"
    elif model == "dolphin-2.1-mistral-7b":
        llm_wrapper = dolphin.Dolphin21MistralWrapper()
    elif model == "dolphin-2.1-mistral-7b-grammar":
        llm_wrapper = dolphin.Dolphin21MistralWrapper(include_opening_brace_in_prefix=False)
        # grammar_name = "json"
        grammar_name = "json_func_calls_with_inner_thoughts"
    elif model == "zephyr-7B-alpha" or model == "zephyr-7B-beta":
        llm_wrapper = zephyr.ZephyrMistralInnerMonologueWrapper()
    elif model == "zephyr-7B-alpha-grammar" or model == "zephyr-7B-beta-grammar":
        llm_wrapper = zephyr.ZephyrMistralInnerMonologueWrapper(include_opening_brace_in_prefix=False)
        # grammar_name = "json"
        grammar_name = "json_func_calls_with_inner_thoughts"
    else:
        # Warn the user that we're using the fallback
        if not has_shown_warning:
            print(
                f"Warning: no wrapper specified for local LLM, using the default wrapper (you can remove this warning by specifying the wrapper with --model)"
            )
            has_shown_warning = True
        if HOST_TYPE in ["koboldcpp", "llamacpp", "webui"]:
            # make the default to use grammar
            llm_wrapper = DEFAULT_WRAPPER(include_opening_brace_in_prefix=False)
            # grammar_name = "json"
            grammar_name = "json_func_calls_with_inner_thoughts"
        else:
            llm_wrapper = DEFAULT_WRAPPER()

    if grammar_name is not None and HOST_TYPE not in ["koboldcpp", "llamacpp", "webui"]:
        print(f"Warning: grammars are currently only supported when using llama.cpp as the MemGPT local LLM backend")

    # First step: turn the message sequence into a prompt that the model expects
    try:
        prompt = llm_wrapper.chat_completion_to_prompt(messages, functions)
        if DEBUG:
            print(prompt)
    except Exception as e:
        raise LocalLLMError(
            f"Failed to convert ChatCompletion messages into prompt string with wrapper {str(llm_wrapper)} - error: {str(e)}"
        )

    try:
        if HOST_TYPE == "webui":
            result = get_webui_completion(prompt, context_window, grammar=grammar_name)
        elif HOST_TYPE == "lmstudio":
            result = get_lmstudio_completion(prompt, context_window)
        elif HOST_TYPE == "llamacpp":
            result = get_llamacpp_completion(prompt, context_window, grammar=grammar_name)
        elif HOST_TYPE == "koboldcpp":
            result = get_koboldcpp_completion(prompt, context_window, grammar=grammar_name)
        elif HOST_TYPE == "ollama":
            result = get_ollama_completion(prompt, context_window)
        else:
            raise LocalLLMError(
                f"BACKEND_TYPE is not set, please set variable depending on your backend (webui, lmstudio, llamacpp, koboldcpp)"
            )
    except requests.exceptions.ConnectionError as e:
        raise LocalLLMConnectionError(f"Unable to connect to host {HOST}")

    if result is None or result == "":
        raise LocalLLMError(f"Got back an empty response string from {HOST}")
    if DEBUG:
        print(f"Raw LLM output:\n{result}")

    try:
        chat_completion_result = llm_wrapper.output_to_chat_completion_response(result)
        if DEBUG:
            print(json.dumps(chat_completion_result, indent=2))
    except Exception as e:
        raise LocalLLMError(f"Failed to parse JSON from local LLM response - error: {str(e)}")

    # unpack with response.choices[0].message.content
    response = DotDict(
        {
            "model": None,
            "choices": [
                DotDict(
                    {
                        "message": DotDict(chat_completion_result),
                        "finish_reason": "stop",  # TODO vary based on backend response
                    }
                )
            ],
            "usage": DotDict(
                {
                    # TODO fix, actually use real info
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ),
        }
    )
    return response
