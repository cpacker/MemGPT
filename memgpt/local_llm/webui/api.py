import os
from urllib.parse import urljoin
import requests

from .settings import SIMPLE
from ..utils import load_grammar_file, count_tokens

WEBUI_API_SUFFIX = "/v1/completions"
DEBUG = False


def get_webui_completion(endpoint, prompt, context_window, settings=SIMPLE, grammar=None):
    """Compatibility for the new OpenAI API: https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt
    request["truncation_length"] = context_window
    request["max_tokens"] = int(context_window - prompt_tokens)
    request["max_new_tokens"] = int(context_window - prompt_tokens)  # safety backup to "max_tokens", shouldn't matter

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
            result = result["choices"][0]["text"]
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
