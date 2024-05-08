import asyncio
import queue
from collections import deque
from typing import AsyncGenerator, Optional

from memgpt.data_types import Message
from memgpt.interface import AgentInterface
from memgpt.models.chat_completion_response import ChatCompletionChunkResponse
from memgpt.streaming_interface import AgentChunkStreamingInterface
from memgpt.utils import get_utc_time, is_utc_datetime


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

    def __init__(self, multi_step=True):
        #
        self.streaming_mode = False

        self._chunks = deque()
        self._event = asyncio.Event()  # Use an event to notify when chunks are available
        self._active = True  # This should be set to False to stop the generator

        # if multi_step = True, the stream ends when the agent yields
        # if multi_step = False, the stream ends when the step ends
        self.multi_step = multi_step
        self.multi_step_indicator = "[DONE_STEP]"
        self.multi_step_gen_indicator = "[DONE_GEN]"

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
        self._chunks.append(self.multi_step_gen_indicator)
        self._event.set()  # Signal that new data is available

        # if not self.multi_step:
        #     # end the stream
        #     self._active = False
        #     self._event.set()  # Unblock the generator if it's waiting to allow it to complete
        # else:
        #     # signal that a new step has started in the stream
        #     self._chunks.append(self.multi_step_indicator)
        #     self._event.set()  # Signal that new data is available

    def process_chunk(self, chunk: ChatCompletionChunkResponse):
        """Process a streaming chunk from an OpenAI-compatible server.

        Example data from non-streaming response looks like:

        data: {"function_call": "send_message({'message': \"Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?\"})", "date": "2024-02-29T06:07:48.844733+00:00"}

        data: {"assistant_message": "Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?", "date": "2024-02-29T06:07:49.846280+00:00"}

        data: {"function_return": "None", "status": "success", "date": "2024-02-29T06:07:50.847262+00:00"}
        """
        # print("Processed CHUNK:", chunk)

        # Example where we just pass through the raw stream from the underlying OpenAI SSE stream
        # processed_chunk = chunk.model_dump_json(exclude_none=True)

        choice = chunk.choices[0]
        message_delta = choice.delta

        # inner thoughts
        if message_delta.content is not None:
            processed_chunk = {
                "internal_monologue": message_delta.content,
            }
        elif message_delta.tool_calls is not None and len(message_delta.tool_calls) > 0:
            tool_call = message_delta.tool_calls[0]

            tool_call_delta = {}
            if tool_call.id:
                tool_call_delta["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.arguments:
                    tool_call_delta["arguments"] = tool_call.function.arguments
                if tool_call.function.name:
                    tool_call_delta["name"] = tool_call.function.name

            processed_chunk = {
                "function_call": tool_call_delta,
            }
        elif choice.finish_reason is not None:
            # skip if there's a finish
            return
        else:
            raise ValueError(f"Couldn't find delta in chunk: {chunk}")

        processed_chunk["date"] = chunk.created.isoformat()

        self._chunks.append(processed_chunk)
        self._event.set()  # Signal that new data is available

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
        if not self.streaming_mode:

            # create a fake "chunk" of a stream
            processed_chunk = {
                "internal_monologue": msg,
                "date": msg_obj.created_at.isoformat() if msg_obj is not None else get_utc_time().isoformat(),
            }

            self._chunks.append(processed_chunk)
            self._event.set()  # Signal that new data is available

        return

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT uses send_message"""
        return

    def function_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT calls a function"""

        # TODO handle 'function' messages that indicate the start of a function call
        assert msg_obj is not None, "StreamingServerInterface requires msg_obj references for metadata"

        if msg.startswith("Running "):
            if not self.streaming_mode:
                # create a fake "chunk" of a stream
                function_call = msg_obj.tool_calls[0]
                processed_chunk = {
                    "function_call": {
                        "id": function_call.id,
                        "name": function_call.function["name"],
                        "arguments": function_call.function["arguments"],
                    },
                    "date": msg_obj.created_at.isoformat(),
                }

                self._chunks.append(processed_chunk)
                self._event.set()  # Signal that new data is available
                return
            else:
                return
            # msg = msg.replace("Running ", "")
            # new_message = {"function_call": msg}

        elif msg.startswith("Ran "):
            return
            # msg = msg.replace("Ran ", "Function call returned: ")
            # new_message = {"function_call": msg}

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

        self._chunks.append(new_message)
        self._event.set()  # Signal that new data is available

    def step_complete(self):
        """Signal from the agent that one 'step' finished (step = LLM response + tool execution)"""
        if not self.multi_step:
            # end the stream
            self._active = False
            self._event.set()  # Unblock the generator if it's waiting to allow it to complete
        else:
            # signal that a new step has started in the stream
            self._chunks.append(self.multi_step_indicator)
            self._event.set()  # Signal that new data is available

    def step_yield(self):
        """If multi_step, this is the true 'stream_end' function."""
        if self.multi_step:
            # end the stream
            self._active = False
            self._event.set()  # Unblock the generator if it's waiting to allow it to complete

    @staticmethod
    def clear():
        return
