import os
from urllib.parse import urljoin
import requests

from .legacy_settings import SIMPLE
from ..utils import load_grammar_file, count_tokens

WEBUI_API_SUFFIX = "/api/v1/generate"
DEBUG = False


def get_webui_completion(endpoint, prompt, context_window, settings=SIMPLE, grammar=None):
    """See https://github.com/oobabooga/text-generation-webui for instructions on how to run the LLM web server"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt
    request["truncation_length"] = context_window  # assuming mistral 7b

    # Set grammar
    if grammar is not None:
        request["grammar_string"] = load_grammar_file(grammar)

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        URI = urljoin(endpoint.strip("/") + "/", WEBUI_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()
            result = result["results"][0]["text"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the web UI server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    return result
