import os
from urllib.parse import urljoin
import requests

from .settings import SIMPLE
from ..utils import count_tokens
from ...errors import LocalLLMError

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
MODEL_NAME = os.getenv("OLLAMA_MODEL")  # ollama API requires this in the request
OLLAMA_API_SUFFIX = "/api/generate"
DEBUG = False


def get_ollama_completion(prompt, context_window, settings=SIMPLE, grammar=None):
    """See https://github.com/jmorganca/ollama/blob/main/docs/api.md for instructions on how to run the LLM web server"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    if MODEL_NAME is None:
        raise LocalLLMError(f"Error: OLLAMA_MODEL not specified. Set OLLAMA_MODEL to the model you want to run (e.g. 'dolphin2.2-mistral')")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt
    request["model"] = MODEL_NAME

    # Set grammar
    if grammar is not None:
        # request["grammar_string"] = load_grammar_file(grammar)
        raise NotImplementedError(f"Ollama does not support grammars")

    if not HOST.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({HOST}) must begin with http:// or https://")

    try:
        URI = urljoin(HOST.strip("/") + "/", OLLAMA_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()
            result = result["response"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the ollama API server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    return result
