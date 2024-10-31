import asyncio
import json
import traceback
import warnings
from enum import Enum
from typing import AsyncGenerator, Optional, Union

from pydantic import BaseModel

from letta.schemas.usage import LettaUsageStatistics
from letta.server.rest_api.interface import StreamingServerInterface
from letta.server.server import SyncServer

# from letta.orm.user import User
# from letta.orm.utilities import get_db_session

SSE_PREFIX = "data: "
SSE_SUFFIX = "\n\n"
SSE_FINISH_MSG = "[DONE]"  # mimic openai
SSE_ARTIFICIAL_DELAY = 0.1


def sse_formatter(data: Union[dict, str]) -> str:
    """Prefix with 'data: ', and always include double newlines"""
    assert type(data) in [dict, str], f"Expected type dict or str, got type {type(data)}"
    data_str = json.dumps(data, separators=(",", ":")) if isinstance(data, dict) else data
    return f"data: {data_str}\n\n"


async def sse_async_generator(
    generator: AsyncGenerator,
    usage_task: Optional[asyncio.Task] = None,
    finish_message=True,
):
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

        # If we have a usage task, wait for it and send its result
        if usage_task is not None:
            try:
                usage = await usage_task
                # Double-check the type
                if not isinstance(usage, LettaUsageStatistics):
                    raise ValueError(f"Expected LettaUsageStatistics, got {type(usage)}")
                yield sse_formatter({"usage": usage.model_dump()})
            except Exception as e:
                warnings.warn(f"Error getting usage data: {e}")
                yield sse_formatter({"error": "Failed to get usage data"})

    except Exception as e:
        print("stream decoder hit error:", e)
        print(traceback.print_stack())
        yield sse_formatter({"error": "stream decoder encountered an error"})

    finally:
        if finish_message:
            # Signal that the stream is complete
            yield sse_formatter(SSE_FINISH_MSG)


# TODO: why does this double up the interface?
def get_letta_server() -> SyncServer:
    # Check if a global server is already instantiated
    from letta.server.rest_api.app import server

    # assert isinstance(server, SyncServer)
    return server


def get_current_interface() -> StreamingServerInterface:
    return StreamingServerInterface
