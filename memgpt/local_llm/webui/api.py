import os
from urllib.parse import urljoin
import requests
import tiktoken

from .settings import SIMPLE

from ...constants import LLM_MAX_TOKENS

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
WEBUI_API_SUFFIX = "/api/v1/generate"
DEBUG = False


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def get_webui_completion(prompt, settings=SIMPLE):
    """See https://github.com/oobabooga/text-generation-webui for instructions on how to run the LLM web server"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > LLM_MAX_TOKENS:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {LLM_MAX_TOKENS} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt

    if not HOST.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({HOST}) must begin with http:// or https://")

    try:
        URI = urljoin(HOST.strip("/") + "/", WEBUI_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()
            result = result["results"][0]["text"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f"Make sure that the web UI server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    return result
