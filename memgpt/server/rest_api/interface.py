import asyncio
import queue
from collections import deque
from typing import AsyncGenerator, Optional

from memgpt.data_types import Message
from memgpt.interface import AgentInterface
from memgpt.models.chat_completion_response import ChatCompletionChunkResponse
from memgpt.streaming_interface import AgentChunkStreamingInterface
from memgpt.utils import is_utc_datetime


class QueuingInterface(AgentInterface):
    """Messages are queued inside an internal buffer and manually flushed"""

    def __init__(self, debug=True):
        self.buffer = queue.Queue()
        self.debug = debug

    def to_list(self):
        """Convert queue to a list (empties it out at the same time)"""
        items = []
        while not self.buffer.empty():
            try:
                items.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        if len(items) > 1 and items[-1] == "STOP":
            items.pop()
        return items

    def clear(self):
        """Clear all messages from the queue."""
        with self.buffer.mutex:
            # Empty the queue
            self.buffer.queue.clear()

    async def message_generator(self):
        while True:
            if not self.buffer.empty():
                message = self.buffer.get()
                if message == "STOP":
                    break
                # yield message | {"date": datetime.now(tz=pytz.utc).isoformat()}
                yield message
            else:
                await asyncio.sleep(0.1)  # Small sleep to prevent a busy loop

    def step_yield(self):
        """Enqueue a special stop message"""
        self.buffer.put("STOP")

    def error(self, error: str):
        """Enqueue a special stop message"""
        self.buffer.put({"internal_error": error})
        self.buffer.put("STOP")

    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        """Handle reception of a user message"""
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None) -> None:
        """Handle the agent's internal monologue"""
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

        new_message = {"internal_monologue": msg}

        # add extra metadata
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()

        self.buffer.put(new_message)

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None) -> None:
        """Handle the agent sending a message"""
        # assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"

        if self.debug:
            print(msg)
            if msg_obj is not None:
                print(vars(msg_obj))
                print(msg_obj.created_at.isoformat())

        new_message = {"assistant_message": msg}

        # add extra metadata
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()
        else:
            # FIXME this is a total hack
            assert self.buffer.qsize() > 1, "Tried to reach back to grab function call data, but couldn't find a buffer message."
            # TODO also should not be accessing protected member here

            new_message["id"] = self.buffer.queue[-1]["id"]
            # assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = self.buffer.queue[-1]["date"]

        self.buffer.put(new_message)

    def function_message(self, msg: str, msg_obj: Optional[Message] = None, include_ran_messages: bool = False) -> None:
        """Handle the agent calling a function"""
        # TODO handle 'function' messages that indicate the start of a function call
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"

        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

        if msg.startswith("Running "):
            msg = msg.replace("Running ", "")
            new_message = {"function_call": msg}

        elif msg.startswith("Ran "):
            if not include_ran_messages:
                return
            msg = msg.replace("Ran ", "Function call returned: ")
            new_message = {"function_call": msg}

        elif msg.startswith("Success: "):
            msg = msg.replace("Success: ", "")
            new_message = {"function_return": msg, "status": "success"}

        elif msg.startswith("Error: "):
            msg = msg.replace("Error: ", "")
            new_message = {"function_return": msg, "status": "error"}

        else:
            # NOTE: generic, should not happen
            new_message = {"function_message": msg}

        # add extra metadata
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()

        self.buffer.put(new_message)


class StreamingServerInterface(AgentChunkStreamingInterface):
    """Maintain a generator that is a proxy for self.process_chunk()

    Usage:
    - The main POST SSE code that launches the streaming request
      will call .process_chunk with each incoming stream (as a handler)
    -

    NOTE: this interface is SINGLE THREADED, and meant to be used
    with a single agent. A multi-agent implementation of this interface
    should maintain multiple generators and index them with the request ID
    """

    def __init__(self):
        self._chunks = deque()
        self._event = asyncio.Event()  # Use an event to notify when chunks are available
        self._active = True  # This should be set to False to stop the generator

    async def _create_generator(self) -> AsyncGenerator:
        """An asynchronous generator that yields chunks as they become available."""
        while self._active:
            # Wait until there is an item in the deque or the stream is deactivated
            await self._event.wait()

            while self._chunks:
                yield self._chunks.popleft()

            # Reset the event until a new item is pushed
            self._event.clear()

    def stream_start(self):
        """Initialize streaming by activating the generator and clearing any old chunks."""
        if not self._active:
            self._active = True
            self._chunks.clear()
            self._event.clear()

    def stream_end(self):
        """Clean up the stream by deactivating and clearing chunks."""
        self._active = False
        self._event.set()  # Unblock the generator if it's waiting to allow it to complete

    def process_chunk(self, chunk: ChatCompletionChunkResponse):
        """Process a streaming chunk from an OpenAI-compatible server."""
        print("Processed CHUNK:", chunk)
        self._chunks.append(chunk.model_dump_json(exclude_none=True))
        self._event.set()  # Signal that new data is available

        # self._chunks.append(chunk.model_dump_json())
        # if self._waiter and not self._waiter.done():
        # self._waiter.set_result(None)

    def get_generator(self) -> AsyncGenerator:
        """Get the generator that yields processed chunks."""
        if not self._active:
            # If the stream is not active, don't return a generator that would produce values
            raise StopIteration("The stream has not been started or has been ended.")
        return self._create_generator()

    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT receives a user message"""
        return

    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT generates some internal monologue"""
        return

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT uses send_message"""
        return

    def function_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT calls a function"""
        return

    def step_yield(self):
        return
