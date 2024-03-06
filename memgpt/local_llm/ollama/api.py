from urllib.parse import urljoin
import requests


from memgpt.local_llm.settings.settings import get_completions_settings
from memgpt.local_llm.utils import post_json_auth_request
from memgpt.utils import count_tokens
from memgpt.errors import LocalLLMError


OLLAMA_API_SUFFIX = "/api/generate"


def get_ollama_completion(endpoint, auth_type, auth_key, model, prompt, context_window, grammar=None):
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
    # https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    settings = get_completions_settings()
    settings.update(
        {
            # specific naming for context length
            "num_ctx": context_window,
        }
    )

    # https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-completion
    request = {
        ## base parameters
        "model": model,
        "prompt": prompt,
        # "images": [],  # TODO eventually support
        ## advanced parameters
        # "format": "json",  # TODO eventually support
        "stream": False,
        "options": settings,
        "raw": True,  # no prompt formatting
        # "raw mode does not support template, system, or context"
        # "system": "",  # no prompt formatting
        # "template": "{{ .Prompt }}",  # no prompt formatting
        # "context": None,  # no memory via prompt formatting
    }

    # Set grammar
    if grammar is not None:
        # request["grammar_string"] = load_grammar_file(grammar)
        raise NotImplementedError(f"Ollama does not support grammars")

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        URI = urljoin(endpoint.strip("/") + "/", OLLAMA_API_SUFFIX.strip("/"))
        response = post_json_auth_request(uri=URI, json_payload=request, auth_type=auth_type, auth_key=auth_key)
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
