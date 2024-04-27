import json
from abc import ABC, abstractmethod
from typing import List, Optional

# from colorama import Fore, Style, init
from rich.console import Console
from rich.live import Live
from rich.markup import escape

from memgpt.data_types import Message
from memgpt.interface import CLIInterface
from memgpt.models.chat_completion_response import (
    ChatCompletionChunkResponse,
    ChatCompletionResponse,
)

# init(autoreset=True)

# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal

STRIP_UI = False


class AgentChunkStreamingInterface(ABC):
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


class StreamingCLIInterface(AgentChunkStreamingInterface):
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


class AgentRefreshStreamingInterface(ABC):
    """Same as the ChunkStreamingInterface, but

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
    def process_refresh(self, response: ChatCompletionResponse):
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

    @abstractmethod
    def toggle_streaming(self, on: bool):
        """Toggle streaming on/off (off = regular CLI interface)"""
        raise NotImplementedError


class StreamingRefreshCLIInterface(AgentRefreshStreamingInterface):
    """Version of the CLI interface that attaches to a stream generator and refreshes a render of the message at every step.

    We maintain the partial message state in the interface state, and on each
    process chunk we:
        (1) update the partial message state,
        (2) refresh/rewrite the state to the screen.
    """

    nonstreaming_interface = CLIInterface

    def __init__(self, fancy: bool = True, separate_send_message: bool = True, disable_inner_mono_call: bool = True):
        """Initialize the streaming CLI interface state."""
        self.console = Console()

        # Using `Live` with `refresh_per_second` parameter to limit the refresh rate, avoiding excessive updates
        self.live = Live("", console=self.console, refresh_per_second=10)
        # self.live.start()  # Start the Live display context and keep it running

        # Use italics / emoji?
        self.fancy = fancy

        self.streaming = True
        self.separate_send_message = separate_send_message
        self.disable_inner_mono_call = disable_inner_mono_call

    def toggle_streaming(self, on: bool):
        self.streaming = on
        if on:
            self.separate_send_message = True
            self.disable_inner_mono_call = True
        else:
            self.separate_send_message = False
            self.disable_inner_mono_call = False

    def update_output(self, content: str):
        """Update the displayed output with new content."""
        # We use the `Live` object's update mechanism to refresh content without clearing the console
        if not self.fancy:
            content = escape(content)
        self.live.update(self.console.render_str(content), refresh=True)

    def process_refresh(self, response: ChatCompletionResponse):
        """Process the response to rewrite the current output buffer."""
        if not response.choices:
            self.update_output("💭 [italic]...[/italic]")
            return  # Early exit if there are no choices

        choice = response.choices[0]
        inner_thoughts = choice.message.content if choice.message.content else ""
        tool_calls = choice.message.tool_calls if choice.message.tool_calls else []

        if self.fancy:
            message_string = f"💭 [italic]{inner_thoughts}[/italic]" if inner_thoughts else ""
        else:
            message_string = "[inner thoughts] " + inner_thoughts if inner_thoughts else ""

        if tool_calls:
            function_call = tool_calls[0].function
            function_name = function_call.name  # Function name, can be an empty string
            function_args = function_call.arguments  # Function arguments, can be an empty string
            if message_string:
                message_string += "\n"
            # special case here for send_message
            if self.separate_send_message and function_name == "send_message":
                try:
                    message = json.loads(function_args)["message"]
                except:
                    prefix = '{\n  "message": "'
                    if len(function_args) < len(prefix):
                        message = "..."
                    elif function_args.startswith(prefix):
                        message = function_args[len(prefix) :]
                    else:
                        message = function_args
                message_string += f"🤖 [bold yellow]{message}[/bold yellow]"
            else:
                message_string += f"{function_name}({function_args})"

        self.update_output(message_string)

    def stream_start(self):
        if self.streaming:
            print()
            self.live.start()  # Start the Live display context and keep it running
            self.update_output("💭 [italic]...[/italic]")

    def stream_end(self):
        if self.streaming:
            if self.live.is_started:
                self.live.stop()
                print()
                self.live = Live("", console=self.console, refresh_per_second=10)

    @staticmethod
    def important_message(msg: str):
        StreamingCLIInterface.nonstreaming_interface.important_message(msg)

    @staticmethod
    def warning_message(msg: str):
        StreamingCLIInterface.nonstreaming_interface.warning_message(msg)

    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None):
        if self.disable_inner_mono_call:
            return
        StreamingCLIInterface.nonstreaming_interface.internal_monologue(msg, msg_obj)

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        if self.separate_send_message:
            return
        StreamingCLIInterface.nonstreaming_interface.assistant_message(msg, msg_obj)

    @staticmethod
    def memory_message(msg: str, msg_obj: Optional[Message] = None):
        StreamingCLIInterface.nonstreaming_interface.memory_message(msg, msg_obj)

    @staticmethod
    def system_message(msg: str, msg_obj: Optional[Message] = None):
        StreamingCLIInterface.nonstreaming_interface.system_message(msg, msg_obj)

    @staticmethod
    def user_message(msg: str, msg_obj: Optional[Message] = None, raw: bool = False, dump: bool = False, debug: bool = DEBUG):
        StreamingCLIInterface.nonstreaming_interface.user_message(msg, msg_obj)

    @staticmethod
    def function_message(msg: str, msg_obj: Optional[Message] = None, debug: bool = DEBUG):
        StreamingCLIInterface.nonstreaming_interface.function_message(msg, msg_obj)

    @staticmethod
    def print_messages(message_sequence: List[Message], dump=False):
        StreamingCLIInterface.nonstreaming_interface.print_messages(message_sequence, dump)

    @staticmethod
    def print_messages_simple(message_sequence: List[Message]):
        StreamingCLIInterface.nonstreaming_interface.print_messages_simple(message_sequence)

    @staticmethod
    def print_messages_raw(message_sequence: List[Message]):
        StreamingCLIInterface.nonstreaming_interface.print_messages_raw(message_sequence)

    @staticmethod
    def step_yield():
        pass
