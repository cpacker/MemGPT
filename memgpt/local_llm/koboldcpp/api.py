import os
from urllib.parse import urljoin
import requests

from memgpt.local_llm.settings.settings import get_completions_settings
from memgpt.local_llm.utils import load_grammar_file, count_tokens

KOBOLDCPP_API_SUFFIX = "/api/v1/generate"


def get_koboldcpp_completion(endpoint, prompt, context_window, grammar=None):
    """See https://lite.koboldai.net/koboldcpp_api for API spec"""
    from memgpt.utils import printd

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    settings = get_completions_settings()
    request = settings
    request["prompt"] = prompt
    request["max_context_length"] = context_window

    # Set grammar
    if grammar is not None:
        request["grammar"] = load_grammar_file(grammar)

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        # NOTE: llama.cpp server returns the following when it's out of context
        # curl: (52) Empty reply from server
        URI = urljoin(endpoint.strip("/") + "/", KOBOLDCPP_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result_full = response.json()
            printd(f"JSON API response:\n{result_full}")
            result = result_full["results"][0]["text"]
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the koboldcpp server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    # Pass usage statistics back to main thread
    # These are used to compute memory warning messages
    # KoboldCpp doesn't return anything?
    # https://lite.koboldai.net/koboldcpp_api#/v1/post_v1_generate
    completion_tokens = None
    total_tokens = prompt_tokens + completion_tokens if completion_tokens is not None else None
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return result, usage
