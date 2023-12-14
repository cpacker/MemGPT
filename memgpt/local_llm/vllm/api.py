import os
from urllib.parse import urljoin
import requests

from ..utils import load_grammar_file, count_tokens

WEBUI_API_SUFFIX = "/v1/completions"


def get_vllm_completion(endpoint, model, prompt, context_window, user, settings={}, grammar=None):
    """https://github.com/vllm-project/vllm/blob/main/examples/api_client.py"""
    from memgpt.utils import printd

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt
    request["max_tokens"] = int(context_window - prompt_tokens)
    request["stream"] = False
    request["user"] = user

    # currently hardcoded, since we are only supporting one model with the hosted endpoint
    request["model"] = model

    # Set grammar
    if grammar is not None:
        raise NotImplementedError
        request["grammar_string"] = load_grammar_file(grammar)

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Endpoint ({endpoint}) must begin with http:// or https://")

    try:
        URI = urljoin(endpoint.strip("/") + "/", WEBUI_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result_full = response.json()
            printd(f"JSON API response:\n{result_full}")
            result = result_full["choices"][0]["text"]
            usage = result_full.get("usage", None)
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the vLLM server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    # Pass usage statistics back to main thread
    # These are used to compute memory warning messages
    completion_tokens = usage.get("completion_tokens", None) if usage is not None else None
    total_tokens = prompt_tokens + completion_tokens if completion_tokens is not None else None
    usage = {
        "prompt_tokens": prompt_tokens,  # can grab from usage dict, but it's usually wrong (set to 0)
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return result, usage
