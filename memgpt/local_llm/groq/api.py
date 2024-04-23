from typing import Tuple
from urllib.parse import urljoin

from memgpt.local_llm.settings.settings import get_completions_settings
from memgpt.local_llm.utils import post_json_auth_request
from memgpt.utils import count_tokens

API_CHAT_SUFFIX = "/v1/chat/completions"
# LMSTUDIO_API_COMPLETIONS_SUFFIX = "/v1/completions"


def get_groq_completion(endpoint: str, auth_type: str, auth_key: str, model: str, prompt: str, context_window: int) -> Tuple[str, dict]:
    """TODO no support for function calling OR raw completions, so we need to route the request into /chat/completions instead"""
    from memgpt.utils import printd

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    settings = get_completions_settings()
    settings.update(
        {
            # see https://console.groq.com/docs/text-chat, supports:
            # "temperature": ,
            # "max_tokens": ,
            # "top_p",
            # "stream",
            # "stop",
            # Groq only allows 4 stop tokens
            "stop": [
                "\nUSER",
                "\nASSISTANT",
                "\nFUNCTION",
                # "\nFUNCTION RETURN",
                # "<|im_start|>",
                # "<|im_end|>",
                # "<|im_sep|>",
                # # airoboros specific
                # "\n### ",
                # # '\n' +
                # # '</s>',
                # # '<|',
                # "\n#",
                # # "\n\n\n",
                # # prevent chaining function calls / multi json objects / run-on generations
                # # NOTE: this requires the ability to patch the extra '}}' back into the prompt
                "  }\n}\n",
            ]
        }
    )

    URI = urljoin(endpoint.strip("/") + "/", API_CHAT_SUFFIX.strip("/"))

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings
    request["model"] = model
    request["max_tokens"] = context_window
    # NOTE: Hack for chat/completion-only endpoints: put the entire completion string inside the first message
    message_structure = [{"role": "user", "content": prompt}]
    request["messages"] = message_structure

    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({endpoint}) must begin with http:// or https://")

    try:
        response = post_json_auth_request(uri=URI, json_payload=request, auth_type=auth_type, auth_key=auth_key)
        if response.status_code == 200:
            result_full = response.json()
            printd(f"JSON API response:\n{result_full}")
            result = result_full["choices"][0]["message"]["content"]
            usage = result_full.get("usage", None)
        else:
            # Example error: msg={"error":"Context length exceeded. Tokens in context: 8000, Context length: 8000"}
            if "context length" in str(response.text).lower():
                # "exceeds context length" is what appears in the LM Studio error message
                # raise an alternate exception that matches OpenAI's message, which is "maximum context length"
                raise Exception(f"Request exceeds maximum context length (code={response.status_code}, msg={response.text}, URI={URI})")
            else:
                raise Exception(
                    f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                    + f" Make sure that the inference server is running and reachable at {URI}."
                )
    except:
        # TODO handle gracefully
        raise

    # Pass usage statistics back to main thread
    # These are used to compute memory warning messages
    completion_tokens = usage.get("completion_tokens", None) if usage is not None else None
    total_tokens = prompt_tokens + completion_tokens if completion_tokens is not None else None
    usage = {
        "prompt_tokens": prompt_tokens,  # can grab from usage dict, but it's usually wrong (set to 0)
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return result, usage
