import os
from urllib.parse import urljoin
import requests

from .settings import SIMPLE
from ..utils import count_tokens

LMSTUDIO_API_CHAT_SUFFIX = "/v1/chat/completions"
LMSTUDIO_API_COMPLETIONS_SUFFIX = "/v1/completions"
DEBUG = False


def get_lmstudio_completion(endpoint, prompt, context_window, settings=SIMPLE, api="chat"):
    """Based on the example for using LM Studio as a backend from https://github.com/lmstudio-ai/examples/tree/main/Hello%2C%20world%20-%20OpenAI%20python%20client"""
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["max_tokens"] = context_window

    if api == "chat":
        # Uses the ChatCompletions API style
        # Seems to work better, probably because it's applying some extra settings under-the-hood?
        URI = urljoin(endpoint.strip("/") + "/", LMSTUDIO_API_CHAT_SUFFIX.strip("/"))
        message_structure = [{"role": "user", "content": prompt}]
        request["messages"] = message_structure
    elif api == "completions":
        # Uses basic string completions (string in, string out)
        # Does not work as well as ChatCompletions for some reason
        URI = urljoin(endpoint.strip("/") + "/", LMSTUDIO_API_COMPLETIONS_SUFFIX.strip("/"))
        request["prompt"] = prompt
    else:
        raise ValueError(api)

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()
            if api == "chat":
                result = result["choices"][0]["message"]["content"]
            elif api == "completions":
                result = result["choices"][0]["text"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            # Example error: msg={"error":"Context length exceeded. Tokens in context: 8000, Context length: 8000"}
            if "context length" in str(response.text).lower():
                # "exceeds context length" is what appears in the LM Studio error message
                # raise an alternate exception that matches OpenAI's message, which is "maximum context length"
                raise Exception(f"Request exceeds maximum context length (code={response.status_code}, msg={response.text}, URI={URI})")
            else:
                raise Exception(
                    f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                    + f" Make sure that the LM Studio local inference server is running and reachable at {URI}."
                )
    except:
        # TODO handle gracefully
        raise

    return result
