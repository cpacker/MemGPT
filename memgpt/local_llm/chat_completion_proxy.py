"""Key idea: create drop-in replacement for agent's ChatCompletion call that runs on an OpenLLM backend"""

import os
import requests
import json

from .webui.api import get_webui_completion
from .webui.legacy_api import get_webui_completion as get_webui_completion_legacy
from .lmstudio.api import get_lmstudio_completion
from .llamacpp.api import get_llamacpp_completion
from .koboldcpp.api import get_koboldcpp_completion
from .ollama.api import get_ollama_completion
from .vllm.api import get_vllm_completion
from .llm_chat_completion_wrappers import airoboros, dolphin, zephyr, simple_summary_wrapper
from .constants import DEFAULT_WRAPPER
from .utils import DotDict, get_available_wrappers
from ..prompts.gpt_summarize import SYSTEM as SUMMARIZE_SYSTEM_MESSAGE
from ..errors import LocalLLMConnectionError, LocalLLMError

endpoint = os.getenv("OPENAI_API_BASE")
endpoint_type = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
DEBUG = False
# DEBUG = True

has_shown_warning = False


def get_chat_completion(
    model,  # no model required (except for Ollama), since the model is fixed to whatever you set in your own backend
    messages,
    functions=None,
    function_call="auto",
    context_window=None,
    # required
    wrapper=None,
    endpoint=None,
    endpoint_type=None,
):
    assert context_window is not None, "Local LLM calls need the context length to be explicitly set"
    assert endpoint is not None, "Local LLM calls need the endpoint (eg http://localendpoint:1234) to be explicitly set"
    assert endpoint_type is not None, "Local LLM calls need the endpoint type (eg webui) to be explicitly set"
    global has_shown_warning
    grammar_name = None

    if function_call != "auto":
        raise ValueError(f"function_call == {function_call} not supported (auto only)")

    available_wrappers = get_available_wrappers()
    if messages[0]["role"] == "system" and messages[0]["content"].strip() == SUMMARIZE_SYSTEM_MESSAGE.strip():
        # Special case for if the call we're making is coming from the summarizer
        llm_wrapper = simple_summary_wrapper.SimpleSummaryWrapper()
    elif wrapper is None:
        # Warn the user that we're using the fallback
        if not has_shown_warning:
            print(
                f"Warning: no wrapper specified for local LLM, using the default wrapper (you can remove this warning by specifying the wrapper with --wrapper)"
            )
            has_shown_warning = True
        if endpoint_type in ["koboldcpp", "llamacpp", "webui"]:
            # make the default to use grammar
            llm_wrapper = DEFAULT_WRAPPER(include_opening_brace_in_prefix=False)
            # grammar_name = "json"
            grammar_name = "json_func_calls_with_inner_thoughts"
        else:
            llm_wrapper = DEFAULT_WRAPPER()
    elif wrapper not in available_wrappers:
        raise ValueError(f"Could not find requested wrapper '{wrapper} in available wrappers list:\n{available_wrappers}")
    else:
        llm_wrapper = available_wrappers[wrapper]
        if "grammar" in wrapper:
            grammar_name = "json_func_calls_with_inner_thoughts"

    if grammar_name is not None and endpoint_type not in ["koboldcpp", "llamacpp", "webui"]:
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
        if endpoint_type == "webui":
            result = get_webui_completion(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "webui-legacy":
            result = get_webui_completion_legacy(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "lmstudio":
            result = get_lmstudio_completion(endpoint, prompt, context_window)
        elif endpoint_type == "llamacpp":
            result = get_llamacpp_completion(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "koboldcpp":
            result = get_koboldcpp_completion(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "ollama":
            result = get_ollama_completion(endpoint, model, prompt, context_window)
        elif endpoint_type == "vllm":
            result = get_vllm_completion(endpoint, model, prompt, context_window)
        else:
            raise LocalLLMError(
                f"BACKEND_TYPE is not set, please set variable depending on your backend (webui, lmstudio, llamacpp, koboldcpp)"
            )
    except requests.exceptions.ConnectionError as e:
        raise LocalLLMConnectionError(f"Unable to connect to endpoint {endpoint}")

    if result is None or result == "":
        raise LocalLLMError(f"Got back an empty response string from {endpoint}")
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
            "model": model,
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
