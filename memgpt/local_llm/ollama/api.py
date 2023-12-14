import os
from urllib.parse import urljoin
import requests

from .settings import SIMPLE
from ..utils import count_tokens
from ...errors import LocalLLMError

OLLAMA_API_SUFFIX = "/api/generate"


def get_ollama_completion(endpoint, model, prompt, context_window, settings=SIMPLE, grammar=None):
    """See https://github.com/jmorganca/ollama/blob/main/docs/api.md for instructions on how to run the LLM web server"""
    from memgpt.utils import printd

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    if model is None:
        raise LocalLLMError(
            f"Error: model name not specified. Set model in your config to the model you want to run (e.g. 'dolphin2.2-mistral')"
        )

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["prompt"] = prompt
    request["model"] = model
    request["options"]["num_ctx"] = context_window

    # Set grammar
    if grammar is not None:
        # request["grammar_string"] = load_grammar_file(grammar)
        raise NotImplementedError(f"Ollama does not support grammars")

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        URI = urljoin(endpoint.strip("/") + "/", OLLAMA_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            # https://github.com/jmorganca/ollama/blob/main/docs/api.md
            result_full = response.json()
            printd(f"JSON API response:\n{result_full}")
            result = result_full["response"]
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the ollama API server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    # Pass usage statistics back to main thread
    # These are used to compute memory warning messages
    # https://github.com/jmorganca/ollama/blob/main/docs/api.md#response
    completion_tokens = result_full.get("eval_count", None)
    total_tokens = prompt_tokens + completion_tokens if completion_tokens is not None else None
    usage = {
        "prompt_tokens": prompt_tokens,  # can also grab from "prompt_eval_count"
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return result, usage
