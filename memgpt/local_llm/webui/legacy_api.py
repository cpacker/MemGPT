from urllib.parse import urljoin

from memgpt.local_llm.settings.settings import get_completions_settings
from memgpt.local_llm.utils import count_tokens, post_json_auth_request

WEBUI_API_SUFFIX = "/api/v1/generate"


def get_webui_completion(endpoint, auth_type, auth_key, prompt, context_window, grammar=None):
    """See https://github.com/oobabooga/text-generation-webui for instructions on how to run the LLM web server"""
    from memgpt.utils import printd

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    settings = get_completions_settings()
    request = settings
    request["stopping_strings"] = request["stop"]  # alias
    request["max_new_tokens"] = 3072  # random hack?
    request["prompt"] = prompt
    request["truncation_length"] = context_window  # assuming mistral 7b

    # Set grammar
    if grammar is not None:
        request["grammar_string"] = grammar

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        URI = urljoin(endpoint.strip("/") + "/", WEBUI_API_SUFFIX.strip("/"))
        response = post_json_auth_request(uri=URI, json_payload=request, auth_type=auth_type, auth_key=auth_key)
        if response.status_code == 200:
            result_full = response.json()
            printd(f"JSON API response:\n{result_full}")
            result = result_full["results"][0]["text"]
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the web UI server is running and reachable at {URI}."
            )

    except:
        # TODO handle gracefully
        raise

    # TODO correct for legacy
    completion_tokens = None
    total_tokens = prompt_tokens + completion_tokens if completion_tokens is not None else None
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return result, usage
