from abc import ABC, abstractmethod
import json
import re
from typing import List, Optional

from colorama import Fore, Style, init

from memgpt.utils import printd
from memgpt.constants import CLI_WARNING_PREFIX, JSON_LOADS_STRICT
from memgpt.data_types import Message
from memgpt.models.chat_completion_response import ChatCompletionChunkResponse
from memgpt.interface import AgentInterface, CLIInterface

init(autoreset=True)

# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal

STRIP_UI = False


class AgentStreamingInterface(ABC):
    """Interfaces handle MemGPT-related events (observer pattern)

    The 'msg' args provides the scoped message, and the optional Message arg can provide additional metadata.
    """

    @abstractmethod
    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT receives a user message"""
        raise NotImplementedError

    @abstractmethod
    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT generates some internal monologue"""
        raise NotImplementedError

    @abstractmethod
    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT uses send_message"""
        raise NotImplementedError

    @abstractmethod
    def function_message(self, msg: str, msg_obj: Optional[Message] = None):
        """MemGPT calls a function"""
        raise NotImplementedError

    @abstractmethod
    def process_chunk(self, chunk: ChatCompletionChunkResponse):
        """Process a streaming chunk from an OpenAI-compatible server"""
        raise NotImplementedError

    @abstractmethod
    def stream_start(self):
        """Any setup required before streaming begins"""
        raise NotImplementedError

    @abstractmethod
    def stream_end(self):
        """Any cleanup required after streaming ends"""
        raise NotImplementedError


class StreamingCLIInterface(AgentStreamingInterface):
    """Version of the CLI interface that attaches to a stream generator and prints along the way.

    When a chunk is received, we write the delta to the buffer. If the buffer type has changed,
    we write out a newline + set the formatting for the new line.

    The two buffer types are:
      (1) content (inner thoughts)
      (2) tool_calls (function calling)

    NOTE: this assumes that the deltas received in the chunks are in-order, e.g.
    that once 'content' deltas stop streaming, they won't be received again. See notes
    on alternative version of the StreamingCLIInterface that does not have this same problem below:

    An alternative implementation could instead maintain the partial message state, and on each
    process chunk (1) update the partial message state, (2) refresh/rewrite the state to the screen.
    """

    # CLIInterface is static/stateless
    nonstreaming_interface = CLIInterface()

    def __init__(self):
        """The streaming CLI interface state for determining which buffer is currently being written to"""

        self.streaming_buffer_type = None

    def _flush(self):
        pass

    def process_chunk(self, chunk: ChatCompletionChunkResponse):
        assert len(chunk.choices) == 1, chunk

        message_delta = chunk.choices[0].delta

        # Starting a new buffer line
        if not self.streaming_buffer_type:
            assert not (
                message_delta.content is not None and message_delta.tool_calls is not None and len(message_delta.tool_calls)
            ), f"Error: got both content and tool_calls in message stream\n{message_delta}"

            if message_delta.content is not None:
                # Write out the prefix for inner thoughts
                print("Inner thoughts: ", end="", flush=True)
            elif message_delta.tool_calls is not None:
                assert len(message_delta.tool_calls) == 1, f"Error: got more than one tool call in response\n{message_delta}"
                # Write out the prefix for function calling
                print("Calling function: ", end="", flush=True)

        # Potentially switch/flush a buffer line
        else:
            pass

        # Write out the delta
        if message_delta.content is not None:
            if self.streaming_buffer_type and self.streaming_buffer_type != "content":
                print()
            self.streaming_buffer_type = "content"

            # Simple, just write out to the buffer
            print(message_delta.content, end="", flush=True)

        elif message_delta.tool_calls is not None:
            if self.streaming_buffer_type and self.streaming_buffer_type != "tool_calls":
                print()
            self.streaming_buffer_type = "tool_calls"

            assert len(message_delta.tool_calls) == 1, f"Error: got more than one tool call in response\n{message_delta}"
            function_call = message_delta.tool_calls[0].function

            # Slightly more complex - want to write parameters in a certain way (paren-style)
            # function_name(function_args)
            if function_call.name:
                # NOTE: need to account for closing the brace later
                print(f"{function_call.name}(", end="", flush=True)
            if function_call.arguments:
                print(function_call.arguments, end="", flush=True)

    def stream_start(self):
        # should be handled by stream_end(), but just in case
        self.streaming_buffer_type = None

    def stream_end(self):
        if self.streaming_buffer_type is not None:
            # TODO: should have a separate self.tool_call_open_paren flag
            if self.streaming_buffer_type == "tool_calls":
                print(")", end="", flush=True)

            print()  # newline to move the cursor
            self.streaming_buffer_type = None  # reset buffer tracker

    @staticmethod
    def important_message(msg: str):
        StreamingCLIInterface.nonstreaming_interface(msg)

    @staticmethod
    def warning_message(msg: str):
        StreamingCLIInterface.nonstreaming_interface(msg)

    @staticmethod
    def internal_monologue(msg: str, msg_obj: Optional[Message] = None):
        StreamingCLIInterface.nonstreaming_interface(msg, msg_obj)

    @staticmethod
    def assistant_message(msg: str, msg_obj: Optional[Message] = None):
        StreamingCLIInterface.nonstreaming_interface(msg, msg_obj)

    @staticmethod
    def memory_message(msg: str, msg_obj: Optional[Message] = None):
        StreamingCLIInterface.nonstreaming_interface(msg, msg_obj)

    @staticmethod
    def system_message(msg: str, msg_obj: Optional[Message] = None):
        StreamingCLIInterface.nonstreaming_interface(msg, msg_obj)

    @staticmethod
    def user_message(msg: str, msg_obj: Optional[Message] = None, raw: bool = False, dump: bool = False, debug: bool = DEBUG):
        StreamingCLIInterface.nonstreaming_interface(msg, msg_obj)

    @staticmethod
    def function_message(msg: str, msg_obj: Optional[Message] = None, debug: bool = DEBUG):
        StreamingCLIInterface.nonstreaming_interface(msg, msg_obj)

    @staticmethod
    def print_messages(message_sequence: List[Message], dump=False):
        StreamingCLIInterface.nonstreaming_interface(message_sequence, dump)

    @staticmethod
    def print_messages_simple(message_sequence: List[Message]):
        StreamingCLIInterface.nonstreaming_interface.print_messages_simple(message_sequence)

    @staticmethod
    def print_messages_raw(message_sequence: List[Message]):
        StreamingCLIInterface.nonstreaming_interface.print_messages_raw(message_sequence)

    @staticmethod
    def step_yield():
        pass
