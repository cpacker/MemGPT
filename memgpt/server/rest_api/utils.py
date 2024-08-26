import traceback
from enum import Enum
from typing import AsyncGenerator, Generator, Union

from pydantic import BaseModel

# from memgpt.orm.user import User
# from memgpt.orm.utilities import get_db_session
from memgpt.utils import json_dumps

SSE_PREFIX = "data: "
SSE_SUFFIX = "\n\n"
SSE_FINISH_MSG = "[DONE]"  # mimic openai
SSE_ARTIFICIAL_DELAY = 0.1


def sse_formatter(data: Union[dict, str]) -> str:
    """Prefix with 'data: ', and always include double newlines"""
    assert type(data) in [dict, str], f"Expected type dict or str, got type {type(data)}"
    data_str = json_dumps(data) if isinstance(data, dict) else data
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
            if isinstance(chunk, BaseModel):
                chunk = chunk.model_dump()
            elif isinstance(chunk, Enum):
                chunk = str(chunk.value)
            elif not isinstance(chunk, dict):
                chunk = str(chunk)
            yield sse_formatter(chunk)

    except Exception as e:
        print("stream decoder hit error:", e)
        print(traceback.print_stack())
        yield sse_formatter({"error": "stream decoder encountered an error"})

    finally:
        if finish_message:
            # Signal that the stream is complete
            yield sse_formatter(SSE_FINISH_MSG)
