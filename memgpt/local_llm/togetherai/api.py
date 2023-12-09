import os
from urllib.parse import urljoin
import requests

from ..utils import load_grammar_file, count_tokens

API_SUFFIX = "/inference"
DEBUG = False


def get_togetherai_completion(endpoint, model, prompt, context_window, user, settings={}, grammar=None):
    """https://docs.together.ai/docs/inference-rest"""
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
        URI = urljoin(endpoint.strip("/") + "/", API_SUFFIX.strip("/"))
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
