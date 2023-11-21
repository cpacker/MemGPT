import os
from urllib.parse import urljoin
import requests

from ..utils import load_grammar_file, count_tokens

WEBUI_API_SUFFIX = "/completions"
DEBUG = False


def get_vllm_completion(endpoint, model, prompt, context_window, settings={}, grammar=None):
    """https://github.com/vllm-project/vllm/blob/main/examples/api_client.py"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt
    request["max_tokens"] = int(context_window - prompt_tokens)
    request["stream"] = False

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
            result = response.json()
            result = result["choices"][0]["text"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the vLLM server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    return result
