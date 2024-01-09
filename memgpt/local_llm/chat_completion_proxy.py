"""Key idea: create drop-in replacement for agent's ChatCompletion call that runs on an OpenLLM backend"""

import os
import requests
import json

from box import Box

from memgpt.local_llm.webui.api import get_webui_completion
from memgpt.local_llm.webui.legacy_api import get_webui_completion as get_webui_completion_legacy
from memgpt.local_llm.lmstudio.api import get_lmstudio_completion
from memgpt.local_llm.llamacpp.api import get_llamacpp_completion
from memgpt.local_llm.koboldcpp.api import get_koboldcpp_completion
from memgpt.local_llm.ollama.api import get_ollama_completion
from memgpt.local_llm.vllm.api import get_vllm_completion
from memgpt.local_llm.llm_chat_completion_wrappers import simple_summary_wrapper
from memgpt.local_llm.constants import DEFAULT_WRAPPER
from memgpt.local_llm.utils import get_available_wrappers, count_tokens
from memgpt.local_llm.function_parser import patch_function
from memgpt.prompts.gpt_summarize import SYSTEM as SUMMARIZE_SYSTEM_MESSAGE
from memgpt.errors import LocalLLMConnectionError, LocalLLMError
from memgpt.constants import CLI_WARNING_PREFIX

has_shown_warning = False


def get_chat_completion(
    model,  # no model required (except for Ollama), since the model is fixed to whatever you set in your own backend
    messages,
    functions=None,
    function_call="auto",
    context_window=None,
    user=None,
    # required
    wrapper=None,
    endpoint=None,
    endpoint_type=None,
    # optional cleanup
    function_correction=True,
    # extra hints to allow for additional prompt formatting hacks
    # TODO this could alternatively be supported via passing function_call="send_message" into the wrapper
    first_message=False,
):
    from memgpt.utils import printd

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
                f"{CLI_WARNING_PREFIX}no wrapper specified for local LLM, using the default wrapper (you can remove this warning by specifying the wrapper with --model-wrapper)"
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
        print(f"{CLI_WARNING_PREFIX}grammars are currently only supported when using llama.cpp as the MemGPT local LLM backend")

    # First step: turn the message sequence into a prompt that the model expects
    try:
        # if hasattr(llm_wrapper, "supports_first_message") and llm_wrapper.supports_first_message:
        if hasattr(llm_wrapper, "supports_first_message"):
            prompt = llm_wrapper.chat_completion_to_prompt(messages, functions, first_message=first_message)
        else:
            prompt = llm_wrapper.chat_completion_to_prompt(messages, functions)
        printd(prompt)
    except Exception as e:
        raise LocalLLMError(
            f"Failed to convert ChatCompletion messages into prompt string with wrapper {str(llm_wrapper)} - error: {str(e)}"
        )

    try:
        if endpoint_type == "webui":
            result, usage = get_webui_completion(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "webui-legacy":
            result, usage = get_webui_completion_legacy(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "lmstudio":
            result, usage = get_lmstudio_completion(endpoint, prompt, context_window, api="completions")
        elif endpoint_type == "lmstudio-legacy":
            result, usage = get_lmstudio_completion(endpoint, prompt, context_window, api="chat")
        elif endpoint_type == "llamacpp":
            result, usage = get_llamacpp_completion(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "koboldcpp":
            result, usage = get_koboldcpp_completion(endpoint, prompt, context_window, grammar=grammar_name)
        elif endpoint_type == "ollama":
            result, usage = get_ollama_completion(endpoint, model, prompt, context_window)
        elif endpoint_type == "vllm":
            result, usage = get_vllm_completion(endpoint, model, prompt, context_window, user)
        else:
            raise LocalLLMError(
                f"Invalid endpoint type {endpoint_type}, please set variable depending on your backend (webui, lmstudio, llamacpp, koboldcpp)"
            )
    except requests.exceptions.ConnectionError as e:
        raise LocalLLMConnectionError(f"Unable to connect to endpoint {endpoint}")

    if result is None or result == "":
        raise LocalLLMError(f"Got back an empty response string from {endpoint}")
    printd(f"Raw LLM output:\n====\n{result}\n====")

    try:
        if hasattr(llm_wrapper, "supports_first_message") and llm_wrapper.supports_first_message:
            chat_completion_result = llm_wrapper.output_to_chat_completion_response(result, first_message=first_message)
        else:
            chat_completion_result = llm_wrapper.output_to_chat_completion_response(result)
        printd(json.dumps(chat_completion_result, indent=2, ensure_ascii=False))
    except Exception as e:
        raise LocalLLMError(f"Failed to parse JSON from local LLM response - error: {str(e)}")

    # Run through some manual function correction (optional)
    if function_correction:
        chat_completion_result = patch_function(message_history=messages, new_message=chat_completion_result)

    # Fill in potential missing usage information (used for tracking token use)
    if not ("prompt_tokens" in usage and "completion_tokens" in usage and "total_tokens" in usage):
        raise LocalLLMError(f"usage dict in response was missing fields ({usage})")

    if usage["prompt_tokens"] is None:
        printd(f"usage dict was missing prompt_tokens, computing on-the-fly...")
        usage["prompt_tokens"] = count_tokens(prompt)

    # NOTE: we should compute on-the-fly anyways since we might have to correct for errors during JSON parsing
    usage["completion_tokens"] = count_tokens(json.dumps(chat_completion_result, ensure_ascii=False))
    """
    if usage["completion_tokens"] is None:
        printd(f"usage dict was missing completion_tokens, computing on-the-fly...")
        # chat_completion_result is dict with 'role' and 'content'
        # token counter wants a string
        usage["completion_tokens"] = count_tokens(json.dumps(chat_completion_result, ensure_ascii=False))
    """

    # NOTE: this is the token count that matters most
    if usage["total_tokens"] is None:
        printd(f"usage dict was missing total_tokens, computing on-the-fly...")
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    # unpack with response.choices[0].message.content
    response = Box(
        {
            "model": model,
            "choices": [
                {
                    "message": chat_completion_result,
                    # TODO vary 'finish_reason' based on backend response
                    # NOTE if we got this far (parsing worked), then it's probably OK to treat this as a stop
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }
    )
    printd(response)
    return response
