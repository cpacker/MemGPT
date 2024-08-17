import asyncio
import json
import traceback
from typing import AsyncGenerator, Generator, Union

from memgpt.constants import JSON_ENSURE_ASCII

SSE_FINISH_MSG = "[DONE]"  # mimic openai
SSE_ARTIFICIAL_DELAY = 0.1


def sse_formatter(data: Union[dict, str]) -> str:
    """Prefix with 'data: ', and always include double newlines"""
    assert type(data) in [dict, str], f"Expected type dict or str, got type {type(data)}"
    data_str = json.dumps(data, ensure_ascii=JSON_ENSURE_ASCII) if isinstance(data, dict) else data
    return f"data: {data_str}\n\n"


async def sse_generator(generator: Generator[dict, None, None]) -> Generator[str, None, None]:
    """Generator that returns 'data: dict' formatted items, e.g.:

    data: {"id":"chatcmpl-9E0PdSZ2IBzAGlQ3SEWHJ5YwzucSP","object":"chat.completion.chunk","created":1713125205,"model":"gpt-4-0613","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"logprobs":null,"finish_reason":null}]}

    data: {"id":"chatcmpl-9E0PdSZ2IBzAGlQ3SEWHJ5YwzucSP","object":"chat.completion.chunk","created":1713125205,"model":"gpt-4-0613","system_fingerprint":null,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}

    data: [DONE]

    """
    try:
        for msg in generator:
            yield sse_formatter(msg)
            if SSE_ARTIFICIAL_DELAY:
                await asyncio.sleep(SSE_ARTIFICIAL_DELAY)  # Sleep to prevent a tight loop, adjust time as needed
    except Exception as e:
        yield sse_formatter({"error": f"{str(e)}"})
    yield sse_formatter(SSE_FINISH_MSG)  # Signal that the stream is complete


async def sse_async_generator(generator: AsyncGenerator, finish_message=True):
    """
    Wraps a generator for use in Server-Sent Events (SSE), handling errors and ensuring a completion message.

    Args:
    - generator: An asynchronous generator yielding data chunks.

    Yields:
    - Formatted Server-Sent Event strings.
    """
    try:
        async for chunk in generator:
            # yield f"data: {json.dumps(chunk)}\n\n"
            yield sse_formatter(chunk)
    except Exception as e:
        print("stream decoder hit error:", e)
        print(traceback.print_stack())
        yield sse_formatter({"error": "stream decoder encountered an error"})
    finally:
        # yield "data: [DONE]\n\n"
        if finish_message:
            yield sse_formatter(SSE_FINISH_MSG)  # Signal that the stream is complete
