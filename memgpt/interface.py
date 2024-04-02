from abc import ABC, abstractmethod
import json
import re
from typing import List, Optional

from colorama import Fore, Style, init

from memgpt.utils import printd
from memgpt.constants import CLI_WARNING_PREFIX, JSON_LOADS_STRICT
from memgpt.data_types import Message

init(autoreset=True)

# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal

STRIP_UI = False


class AgentInterface(ABC):
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

    # @abstractmethod
    # @staticmethod
    # def print_messages():
    #     raise NotImplementedError

    # @abstractmethod
    # @staticmethod
    # def print_messages_raw():
    #     raise NotImplementedError

    # @abstractmethod
    # @staticmethod
    # def step_yield():
    #     raise NotImplementedError


class CLIInterface(AgentInterface):
    """Basic interface for dumping agent events to the command-line"""

    @staticmethod
    def important_message(msg: str):
        fstr = f"{Fore.MAGENTA}{Style.BRIGHT}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def warning_message(msg: str):
        fstr = f"{Fore.RED}{Style.BRIGHT}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        else:
            print(fstr.format(msg=msg))

    @staticmethod
    def internal_monologue(msg: str, msg_obj: Optional[Message] = None):
        # ANSI escape code for italic is '\x1B[3m'
        fstr = f"\x1B[3m{Fore.LIGHTBLACK_EX}💭 {{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def assistant_message(msg: str, msg_obj: Optional[Message] = None):
        fstr = f"{Fore.YELLOW}{Style.BRIGHT}🤖 {Fore.YELLOW}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def memory_message(msg: str, msg_obj: Optional[Message] = None):
        fstr = f"{Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}🧠 {Fore.LIGHTMAGENTA_EX}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def system_message(msg: str, msg_obj: Optional[Message] = None):
        fstr = f"{Fore.MAGENTA}{Style.BRIGHT}🖥️ [system] {Fore.MAGENTA}{msg}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def user_message(msg: str, msg_obj: Optional[Message] = None, raw: bool = False, dump: bool = False, debug: bool = DEBUG):
        def print_user_message(icon, msg, printf=print):
            if STRIP_UI:
                printf(f"{icon} {msg}")
            else:
                printf(f"{Fore.GREEN}{Style.BRIGHT}{icon} {Fore.GREEN}{msg}{Style.RESET_ALL}")

        def printd_user_message(icon, msg):
            return print_user_message(icon, msg)

        if not (raw or dump or debug):
            # we do not want to repeat the message in normal use
            return

        if isinstance(msg, str):
            if raw:
                printd_user_message("🧑", msg)
                return
            else:
                try:
                    msg_json = json.loads(msg, strict=JSON_LOADS_STRICT)
                except:
                    printd(f"{CLI_WARNING_PREFIX}failed to parse user message into json")
                    printd_user_message("🧑", msg)
                    return
        if msg_json["type"] == "user_message":
            if dump:
                print_user_message("🧑", msg_json["message"])
                return
            msg_json.pop("type")
            printd_user_message("🧑", msg_json)
        elif msg_json["type"] == "heartbeat":
            if debug:
                msg_json.pop("type")
                printd_user_message("💓", msg_json)
            elif dump:
                print_user_message("💓", msg_json)
                return

        elif msg_json["type"] == "system_message":
            msg_json.pop("type")
            printd_user_message("🖥️", msg_json)
        else:
            printd_user_message("🧑", msg_json)

    @staticmethod
    def function_message(msg: str, msg_obj: Optional[Message] = None, debug: bool = DEBUG):

        def print_function_message(icon, msg, color=Fore.RED, printf=print):
            if STRIP_UI:
                printf(f"⚡{icon} [function] {msg}")
            else:
                printf(f"{color}{Style.BRIGHT}⚡{icon} [function] {color}{msg}{Style.RESET_ALL}")

        def printd_function_message(icon, msg, color=Fore.RED):
            return print_function_message(icon, msg, color, printf=(print if debug else printd))

        if isinstance(msg, dict):
            printd_function_message("", msg)
            return

        if msg.startswith("Success"):
            printd_function_message("🟢", msg)
        elif msg.startswith("Error: "):
            printd_function_message("🔴", msg)
        elif msg.startswith("Ran "):
            # NOTE: ignore 'ran' messages that come post-execution
            return
        elif msg.startswith("Running "):
            if debug:
                printd_function_message("", msg)
            else:
                match = re.search(r"Running (\w+)\((.*)\)", msg)
                if match:
                    function_name = match.group(1)
                    function_args = match.group(2)
                    if function_name in ["archival_memory_insert", "archival_memory_search", "core_memory_replace", "core_memory_append"]:
                        if function_name in ["archival_memory_insert", "core_memory_append", "core_memory_replace"]:
                            print_function_message("🧠", f"updating memory with {function_name}")
                        elif function_name == "archival_memory_search":
                            print_function_message("🧠", f"searching memory with {function_name}")
                        try:
                            msg_dict = eval(function_args)
                            if function_name == "archival_memory_search":
                                output = f'\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                                if STRIP_UI:
                                    print(output)
                                else:
                                    print(f"{Fore.RED}{output}{Style.RESET_ALL}")
                            elif function_name == "archival_memory_insert":
                                output = f'\t→ {msg_dict["content"]}'
                                if STRIP_UI:
                                    print(output)
                                else:
                                    print(f"{Style.BRIGHT}{Fore.RED}{output}{Style.RESET_ALL}")
                            else:
                                if STRIP_UI:
                                    print(f'\t {msg_dict["old_content"]}\n\t→ {msg_dict["new_content"]}')
                                else:
                                    print(
                                        f'{Style.BRIGHT}\t{Fore.RED} {msg_dict["old_content"]}\n\t{Fore.GREEN}→ {msg_dict["new_content"]}{Style.RESET_ALL}'
                                    )
                        except Exception as e:
                            printd(str(e))
                            printd(msg_dict)
                    elif function_name in ["conversation_search", "conversation_search_date"]:
                        print_function_message("🧠", f"searching memory with {function_name}")
                        try:
                            msg_dict = eval(function_args)
                            output = f'\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                            if STRIP_UI:
                                print(output)
                            else:
                                print(f"{Fore.RED}{output}{Style.RESET_ALL}")
                        except Exception as e:
                            printd(str(e))
                            printd(msg_dict)
                else:
                    printd(f"{CLI_WARNING_PREFIX}did not recognize function message")
                    printd_function_message("", msg)
        else:
            try:
                msg_dict = json.loads(msg, strict=JSON_LOADS_STRICT)
                if "status" in msg_dict and msg_dict["status"] == "OK":
                    printd_function_message("", str(msg), color=Fore.GREEN)
                else:
                    printd_function_message("", str(msg), color=Fore.RED)
            except Exception:
                print(f"{CLI_WARNING_PREFIX}did not recognize function message {type(msg)} {msg}")
                printd_function_message("", msg)

    @staticmethod
    def print_messages(message_sequence: List[Message], dump=False):
        # rewrite to dict format
        message_sequence = [msg.to_openai_dict() for msg in message_sequence]

        idx = len(message_sequence)
        for msg in message_sequence:
            if dump:
                print(f"[{idx}] ", end="")
                idx -= 1
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                CLIInterface.system_message(content)
            elif role == "assistant":
                # Differentiate between internal monologue, function calls, and messages
                if msg.get("function_call"):
                    if content is not None:
                        CLIInterface.internal_monologue(content)
                    # I think the next one is not up to date
                    # function_message(msg["function_call"])
                    args = json.loads(msg["function_call"].get("arguments"), strict=JSON_LOADS_STRICT)
                    CLIInterface.assistant_message(args.get("message"))
                    # assistant_message(content)
                elif msg.get("tool_calls"):
                    if content is not None:
                        CLIInterface.internal_monologue(content)
                    function_obj = msg["tool_calls"][0].get("function")
                    if function_obj:
                        args = json.loads(function_obj.get("arguments"), strict=JSON_LOADS_STRICT)
                        CLIInterface.assistant_message(args.get("message"))
                else:
                    CLIInterface.internal_monologue(content)
            elif role == "user":
                CLIInterface.user_message(content, dump=dump)
            elif role == "function":
                CLIInterface.function_message(content, debug=dump)
            elif role == "tool":
                CLIInterface.function_message(content, debug=dump)
            else:
                print(f"Unknown role: {content}")

    @staticmethod
    def print_messages_simple(message_sequence: List[Message]):
        # rewrite to dict format
        message_sequence = [msg.to_openai_dict() for msg in message_sequence]

        for msg in message_sequence:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                CLIInterface.system_message(content)
            elif role == "assistant":
                CLIInterface.assistant_message(content)
            elif role == "user":
                CLIInterface.user_message(content, raw=True)
            else:
                print(f"Unknown role: {content}")

    @staticmethod
    def print_messages_raw(message_sequence: List[Message]):
        # rewrite to dict format
        message_sequence = [msg.to_openai_dict() for msg in message_sequence]

        for msg in message_sequence:
            print(msg)

    @staticmethod
    def step_yield():
        pass
