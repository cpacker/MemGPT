import json
from typing import Generator

from memgpt.constants import JSON_ENSURE_ASCII

SSE_FINISH_MSG = "[DONE]"  # mimic openai


def sse_formatter(data: dict) -> str:
    """Prefix with 'data: ', and always include double newlines"""
    return f"data: {json.dumps(data, ensure_ascii=JSON_ENSURE_ASCII)}\n\n"


def sse_generator(generator: Generator[dict, None, None]) -> Generator[str, None, None]:
    """Generator that returns 'data: dict' formatted items, e.g.:

    data: {"id":"chatcmpl-9E0PdSZ2IBzAGlQ3SEWHJ5YwzucSP","object":"chat.completion.chunk","created":1713125205,"model":"gpt-4-0613","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chatcmpl-9E0PdSZ2IBzAGlQ3SEWHJ5YwzucSP","object":"chat.completion.chunk","created":1713125205,"model":"gpt-4-0613","system_fingerprint":null,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}

    data: [DONE]

    """
    try:
        for msg in generator:
            yield sse_formatter(msg)
            # NOTE: do the waiting in the method that yields the sse_generator
            # await asyncio.sleep(1)  # Sleep to prevent a tight loop, adjust time as needed
    except Exception as e:
        yield sse_formatter({"error": f"{str(e)}"})
    yield sse_formatter(SSE_FINISH_MSG)  # Signal that the stream is complete
