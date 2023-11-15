import os
from urllib.parse import urljoin
import requests

from .settings import SIMPLE
from ..utils import load_grammar_file, count_tokens

LLAMACPP_API_SUFFIX = "/completion"
DEBUG = False
# DEBUG = True


def get_llamacpp_completion(endpoint, prompt, context_window, grammar=None, settings=SIMPLE):
    """See https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md for instructions on how to run the LLM web server"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt

    # Set grammar
    if grammar is not None:
        request["grammar"] = load_grammar_file(grammar)

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        # NOTE: llama.cpp server returns the following when it's out of context
        # curl: (52) Empty reply from server
        URI = urljoin(endpoint.strip("/") + "/", LLAMACPP_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()
            result = result["content"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the llama.cpp server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    return result
