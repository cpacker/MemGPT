import asyncio
import json
import queue
from collections import deque
from datetime import datetime
from typing import AsyncGenerator, Literal, Optional, Union

from memgpt.interface import AgentInterface
from memgpt.schemas.enums import MessageStreamStatus
from memgpt.schemas.memgpt_message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallDelta,
    FunctionCallMessage,
    FunctionReturn,
    InternalMonologue,
    LegacyFunctionCallMessage,
    LegacyMemGPTMessage,
    MemGPTMessage,
)
from memgpt.schemas.message import Message
from memgpt.schemas.openai.chat_completion_response import ChatCompletionChunkResponse
from memgpt.streaming_interface import AgentChunkStreamingInterface
from memgpt.utils import is_utc_datetime


class QueuingInterface(AgentInterface):
    """Messages are queued inside an internal buffer and manually flushed"""

    def __init__(self, debug=True):
        self.buffer = queue.Queue()
        self.debug = debug

    def _queue_push(self, message_api: Union[str, dict], message_obj: Union[Message, None]):
        """Wrapper around self.buffer.queue.put() that ensures the types are safe

        Data will be in the format: {
            "message_obj": ...
            "message_string": ...
        }
        """

        # Check the string first

        if isinstance(message_api, str):
            # check that it's the stop word
            if message_api == "STOP":
                assert message_obj is None
                self.buffer.put(
                    {
                        "message_api": message_api,
                        "message_obj": None,
                    }
                )
            else:
                raise ValueError(f"Unrecognized string pushed to buffer: {message_api}")

        elif isinstance(message_api, dict):
            # check if it's the error message style
            if len(message_api.keys()) == 1 and "internal_error" in message_api:
                assert message_obj is None
                self.buffer.put(
                    {
                        "message_api": message_api,
                        "message_obj": None,
                    }
                )
            else:
                assert message_obj is not None, message_api
                self.buffer.put(
                    {
                        "message_api": message_api,
                        "message_obj": message_obj,
                    }
                )

        else:
            raise ValueError(f"Unrecognized type pushed to buffer: {type(message_api)}")

    def to_list(self, style: Literal["obj", "api"] = "obj"):
        """Convert queue to a list (empties it out at the same time)"""
        items = []
        while not self.buffer.empty():
            try:
                # items.append(self.buffer.get_nowait())
                item_to_push = self.buffer.get_nowait()
                if style == "obj":
                    if item_to_push["message_obj"] is not None:
                        items.append(item_to_push["message_obj"])
                elif style == "api":
                    items.append(item_to_push["message_api"])
                else:
                    raise ValueError(style)
            except queue.Empty:
                break
        if len(items) > 1 and items[-1] == "STOP":
            items.pop()

        # If the style is "obj", then we need to deduplicate any messages
        # Filter down items for duplicates based on item.id
        if style == "obj":
            seen_ids = set()
            unique_items = []
            for item in reversed(items):
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    unique_items.append(item)
            items = list(reversed(unique_items))

        return items

    def clear(self):
        """Clear all messages from the queue."""
        with self.buffer.mutex:
            # Empty the queue
            self.buffer.queue.clear()

    async def message_generator(self, style: Literal["obj", "api"] = "obj"):
        while True:
            if not self.buffer.empty():
                message = self.buffer.get()
                message_obj = message["message_obj"]
                message_api = message["message_api"]

                if message_api == "STOP":
                    break

                # yield message
                if style == "obj":
                    yield message_obj
                elif style == "api":
                    yield message_api
                else:
                    raise ValueError(style)

            else:
                await asyncio.sleep(0.1)  # Small sleep to prevent a busy loop

    def step_yield(self):
        """Enqueue a special stop message"""
        self._queue_push(message_api="STOP", message_obj=None)

    @staticmethod
    def step_complete():
        pass

    def error(self, error: str):
        """Enqueue a special stop message"""
        self._queue_push(message_api={"internal_error": error}, message_obj=None)
        self._queue_push(message_api="STOP", message_obj=None)

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

        self._queue_push(message_api=new_message, message_obj=msg_obj)

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

            new_message["id"] = self.buffer.queue[-1]["message_api"]["id"]
            # assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = self.buffer.queue[-1]["message_api"]["date"]

            msg_obj = self.buffer.queue[-1]["message_obj"]

        self._queue_push(message_api=new_message, message_obj=msg_obj)

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

        self._queue_push(message_api=new_message, message_obj=msg_obj)


class FunctionArgumentsStreamHandler:
    """State machine that can process a stream of"""

    def __init__(self, json_key="message"):
        self.json_key = json_key
        self.reset()

    def reset(self):
        self.in_message = False
        self.key_buffer = ""
        self.accumulating = False
        self.message_started = False

    def process_json_chunk(self, chunk: str) -> Optional[str]:
        """Process a chunk from the function arguments and return the plaintext version"""

        # Use strip to handle only leading and trailing whitespace in control structures
        if self.accumulating:
            clean_chunk = chunk.strip()
            if self.json_key in self.key_buffer:
                if ":" in clean_chunk:
                    self.in_message = True
                    self.accumulating = False
                    return None
            self.key_buffer += clean_chunk
            return None

        if self.in_message:
            if chunk.strip() == '"' and self.message_started:
                self.in_message = False
                self.message_started = False
                return None
            if not self.message_started and chunk.strip() == '"':
                self.message_started = True
                return None
            if self.message_started:
                if chunk.strip().endswith('"'):
                    self.in_message = False
                    return chunk.rstrip('"\n')
                return chunk

        if chunk.strip() == "{":
            self.key_buffer = ""
            self.accumulating = True
            return None
        if chunk.strip() == "}":
            self.in_message = False
            self.message_started = False
            return None
        return None


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
        # If streaming mode, ignores base interface calls like .assistant_message, etc
        self.streaming_mode = False
        # NOTE: flag for supporting legacy 'stream' flag where send_message is treated specially
        self.nonstreaming_legacy_mode = False
        # If chat completion mode, creates a "chatcompletion-style" stream, but with concepts remapped
        self.streaming_chat_completion_mode = False
        self.streaming_chat_completion_mode_function_name = None  # NOTE: sadly need to track state during stream
        # If chat completion mode, we need a special stream reader to
        # turn function argument to send_message into a normal text stream
        self.streaming_chat_completion_json_reader = FunctionArgumentsStreamHandler()

        self._chunks = deque()
        self._event = asyncio.Event()  # Use an event to notify when chunks are available
        self._active = True  # This should be set to False to stop the generator

        # if multi_step = True, the stream ends when the agent yields
        # if multi_step = False, the stream ends when the step ends
        self.multi_step = multi_step
        self.multi_step_indicator = MessageStreamStatus.done_step
        self.multi_step_gen_indicator = MessageStreamStatus.done_generation

        # extra prints
        self.debug = False
        self.timeout = 30

    async def _create_generator(self) -> AsyncGenerator[Union[MemGPTMessage, LegacyMemGPTMessage, MessageStreamStatus], None]:
        """An asynchronous generator that yields chunks as they become available."""
        while self._active:
            try:
                # Wait until there is an item in the deque or the stream is deactivated
                await asyncio.wait_for(self._event.wait(), timeout=self.timeout)  # 30 second timeout
            except asyncio.TimeoutError:
                break  # Exit the loop if we timeout

            while self._chunks:
                yield self._chunks.popleft()

            # Reset the event until a new item is pushed
            self._event.clear()

        # while self._active:
        #     # Wait until there is an item in the deque or the stream is deactivated
        #     await self._event.wait()

        #     while self._chunks:
        #         yield self._chunks.popleft()

        #     # Reset the event until a new item is pushed
        #     self._event.clear()

    def get_generator(self) -> AsyncGenerator:
        """Get the generator that yields processed chunks."""
        if not self._active:
            # If the stream is not active, don't return a generator that would produce values
            raise StopIteration("The stream has not been started or has been ended.")
        return self._create_generator()

    def _push_to_buffer(
        self,
        item: Union[
            # signal on SSE stream status [DONE_GEN], [DONE_STEP], [DONE]
            MessageStreamStatus,
            # the non-streaming message types
            MemGPTMessage,
            LegacyMemGPTMessage,
            # the streaming message types
            ChatCompletionChunkResponse,
        ],
    ):
        """Add an item to the deque"""
        assert self._active, "Generator is inactive"
        assert (
            isinstance(item, MemGPTMessage) or isinstance(item, LegacyMemGPTMessage) or isinstance(item, MessageStreamStatus)
        ), f"Wrong type: {type(item)}"

        self._chunks.append(item)
        self._event.set()  # Signal that new data is available

    def stream_start(self):
        """Initialize streaming by activating the generator and clearing any old chunks."""
        self.streaming_chat_completion_mode_function_name = None

        if not self._active:
            self._active = True
            self._chunks.clear()
            self._event.clear()

    def stream_end(self):
        """Clean up the stream by deactivating and clearing chunks."""
        self.streaming_chat_completion_mode_function_name = None

        if not self.streaming_chat_completion_mode and not self.nonstreaming_legacy_mode:
            self._push_to_buffer(self.multi_step_gen_indicator)

        # self._active = False
        # self._event.set()  # Unblock the generator if it's waiting to allow it to complete

        # if not self.multi_step:
        #     # end the stream
        #     self._active = False
        #     self._event.set()  # Unblock the generator if it's waiting to allow it to complete
        # else:
        #     # signal that a new step has started in the stream
        #     self._chunks.append(self.multi_step_indicator)
        #     self._event.set()  # Signal that new data is available

    def step_complete(self):
        """Signal from the agent that one 'step' finished (step = LLM response + tool execution)"""
        if not self.multi_step:
            # end the stream
            self._active = False
            self._event.set()  # Unblock the generator if it's waiting to allow it to complete
        elif not self.streaming_chat_completion_mode and not self.nonstreaming_legacy_mode:
            # signal that a new step has started in the stream
            self._push_to_buffer(self.multi_step_indicator)

    def step_yield(self):
        """If multi_step, this is the true 'stream_end' function."""
        # if self.multi_step:
        # end the stream
        self._active = False
        self._event.set()  # Unblock the generator if it's waiting to allow it to complete

    @staticmethod
    def clear():
        return

    def _process_chunk_to_memgpt_style(
        self, chunk: ChatCompletionChunkResponse, message_id: str, message_date: datetime
    ) -> Optional[Union[InternalMonologue, FunctionCallMessage]]:
        """
        Example data from non-streaming response looks like:

        data: {"function_call": "send_message({'message': \"Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?\"})", "date": "2024-02-29T06:07:48.844733+00:00"}

        data: {"assistant_message": "Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?", "date": "2024-02-29T06:07:49.846280+00:00"}

        data: {"function_return": "None", "status": "success", "date": "2024-02-29T06:07:50.847262+00:00"}
        """
        choice = chunk.choices[0]
        message_delta = choice.delta

        # inner thoughts
        if message_delta.content is not None:
            processed_chunk = InternalMonologue(
                id=message_id,
                date=message_date,
                internal_monologue=message_delta.content,
            )
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

            processed_chunk = FunctionCallMessage(
                id=message_id,
                date=message_date,
                function_call=FunctionCallDelta(name=tool_call_delta.get("name"), arguments=tool_call_delta.get("arguments")),
            )
        elif choice.finish_reason is not None:
            # skip if there's a finish
            return None
        else:
            raise ValueError(f"Couldn't find delta in chunk: {chunk}")

        return processed_chunk

    def _process_chunk_to_openai_style(self, chunk: ChatCompletionChunkResponse) -> Optional[dict]:
        """Chunks should look like OpenAI, but be remapped from MemGPT-style concepts.

        inner_thoughts are silenced:
          - means that 'content' -> /dev/null
        send_message is a "message"
          - means that tool call to "send_message" should map to 'content'

        TODO handle occurance of multi-step function calling
        TODO handle partial stream of "name" in tool call
        """
        proxy_chunk = chunk.model_copy(deep=True)

        choice = chunk.choices[0]
        message_delta = choice.delta

        # inner thoughts
        if message_delta.content is not None:
            # skip inner monologue
            return None

        # tool call
        elif message_delta.tool_calls is not None and len(message_delta.tool_calls) > 0:
            tool_call = message_delta.tool_calls[0]

            if tool_call.function:

                # Track the function name while streaming
                # If we were previously on a 'send_message', we need to 'toggle' into 'content' mode
                if tool_call.function.name:
                    if self.streaming_chat_completion_mode_function_name is None:
                        self.streaming_chat_completion_mode_function_name = tool_call.function.name
                    else:
                        self.streaming_chat_completion_mode_function_name += tool_call.function.name

                    if tool_call.function.name == "send_message":
                        # early exit to turn into content mode
                        self.streaming_chat_completion_json_reader.reset()
                        return None

                if tool_call.function.arguments:
                    if self.streaming_chat_completion_mode_function_name == "send_message":
                        cleaned_func_args = self.streaming_chat_completion_json_reader.process_json_chunk(tool_call.function.arguments)
                        if cleaned_func_args is None:
                            return None
                        else:
                            # Wipe tool call
                            proxy_chunk.choices[0].delta.tool_calls = None
                            # Replace with 'content'
                            proxy_chunk.choices[0].delta.content = cleaned_func_args

        processed_chunk = proxy_chunk.model_dump(exclude_none=True)

        return processed_chunk

    def process_chunk(self, chunk: ChatCompletionChunkResponse, message_id: str, message_date: datetime):
        """Process a streaming chunk from an OpenAI-compatible server.

        Example data from non-streaming response looks like:

        data: {"function_call": "send_message({'message': \"Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?\"})", "date": "2024-02-29T06:07:48.844733+00:00"}

        data: {"assistant_message": "Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?", "date": "2024-02-29T06:07:49.846280+00:00"}

        data: {"function_return": "None", "status": "success", "date": "2024-02-29T06:07:50.847262+00:00"}
        """
        # print("Processed CHUNK:", chunk)

        # Example where we just pass through the raw stream from the underlying OpenAI SSE stream
        # processed_chunk = chunk.model_dump_json(exclude_none=True)

        if self.streaming_chat_completion_mode:
            # processed_chunk = self._process_chunk_to_openai_style(chunk)
            raise NotImplementedError("OpenAI proxy streaming temporarily disabled")
        else:
            processed_chunk = self._process_chunk_to_memgpt_style(chunk=chunk, message_id=message_id, message_date=message_date)

        if processed_chunk is None:
            return

        self._push_to_buffer(processed_chunk)

    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT receives a user message"""
        return

    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT generates some internal monologue"""
        if not self.streaming_mode:

            # create a fake "chunk" of a stream
            # processed_chunk = {
            #     "internal_monologue": msg,
            #     "date": msg_obj.created_at.isoformat() if msg_obj is not None else get_utc_time().isoformat(),
            #     "id": str(msg_obj.id) if msg_obj is not None else None,
            # }
            processed_chunk = InternalMonologue(
                id=msg_obj.id,
                date=msg_obj.created_at,
                internal_monologue=msg,
            )

            self._push_to_buffer(processed_chunk)

        return

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT uses send_message"""

        # if not self.streaming_mode and self.send_message_special_case:

        #     # create a fake "chunk" of a stream
        #     processed_chunk = {
        #         "assistant_message": msg,
        #         "date": msg_obj.created_at.isoformat() if msg_obj is not None else get_utc_time().isoformat(),
        #         "id": str(msg_obj.id) if msg_obj is not None else None,
        #     }

        #     self._chunks.append(processed_chunk)
        #     self._event.set()  # Signal that new data is available

        return

    def function_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT calls a function"""

        # TODO handle 'function' messages that indicate the start of a function call
        assert msg_obj is not None, "StreamingServerInterface requires msg_obj references for metadata"

        if msg.startswith("Running "):
            if not self.streaming_mode:
                # create a fake "chunk" of a stream
                function_call = msg_obj.tool_calls[0]

                if self.nonstreaming_legacy_mode:
                    # Special case where we want to send two chunks - one first for the function call, then for send_message

                    # Should be in the following legacy style:
                    # data: {
                    #   "function_call": "send_message({'message': 'Chad, ... ask?'})",
                    #   "id": "771748ee-120a-453a-960d-746570b22ee5",
                    #   "date": "2024-06-22T23:04:32.141923+00:00"
                    # }
                    try:
                        func_args = json.loads(function_call.function.arguments)
                    except:
                        func_args = function_call.function.arguments
                    # processed_chunk = {
                    #     "function_call": f"{function_call.function.name}({func_args})",
                    #     "id": str(msg_obj.id),
                    #     "date": msg_obj.created_at.isoformat(),
                    # }
                    processed_chunk = LegacyFunctionCallMessage(
                        id=msg_obj.id,
                        date=msg_obj.created_at,
                        function_call=f"{function_call.function.name}({func_args})",
                    )
                    self._push_to_buffer(processed_chunk)

                    if function_call.function.name == "send_message":
                        try:
                            # processed_chunk = {
                            #     "assistant_message": func_args["message"],
                            #     "id": str(msg_obj.id),
                            #     "date": msg_obj.created_at.isoformat(),
                            # }
                            processed_chunk = AssistantMessage(
                                id=msg_obj.id,
                                date=msg_obj.created_at,
                                assistant_message=func_args["message"],
                            )
                            self._push_to_buffer(processed_chunk)
                        except Exception as e:
                            print(f"Failed to parse function message: {e}")

                else:

                    processed_chunk = FunctionCallMessage(
                        id=msg_obj.id,
                        date=msg_obj.created_at,
                        function_call=FunctionCall(
                            name=function_call.function.name,
                            arguments=function_call.function.arguments,
                        ),
                    )
                    # processed_chunk = {
                    #     "function_call": {
                    #         "name": function_call.function.name,
                    #         "arguments": function_call.function.arguments,
                    #     },
                    #     "id": str(msg_obj.id),
                    #     "date": msg_obj.created_at.isoformat(),
                    # }
                    self._push_to_buffer(processed_chunk)

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
            # new_message = {"function_return": msg, "status": "success"}
            new_message = FunctionReturn(
                id=msg_obj.id,
                date=msg_obj.created_at,
                function_return=msg,
                status="success",
            )

        elif msg.startswith("Error: "):
            msg = msg.replace("Error: ", "")
            # new_message = {"function_return": msg, "status": "error"}
            new_message = FunctionReturn(
                id=msg_obj.id,
                date=msg_obj.created_at,
                function_return=msg,
                status="error",
            )

        else:
            # NOTE: generic, should not happen
            raise ValueError(msg)
            new_message = {"function_message": msg}

        # add extra metadata
        # if msg_obj is not None:
        #     new_message["id"] = str(msg_obj.id)
        #     assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
        #     new_message["date"] = msg_obj.created_at.isoformat()

        self._push_to_buffer(new_message)
