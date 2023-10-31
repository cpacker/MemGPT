import json
import re

from colorama import Fore, Style, init

init(autoreset=True)


# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal


class DummyInterface(object):
    def set_message_list(self, message_list):
        pass

    def internal_monologue(self, msg):
        pass

    def assistant_message(self, msg):
        pass

    def memory_message(self, msg):
        pass

    def system_message(self, msg):
        pass

    def user_message(self, msg, raw=False):
        pass

    def function_message(self, msg):
        pass


class AutoGenInterface(object):
    """AutoGen expects a single action return in its step loop, but MemGPT may take many actions.

    To support AutoGen, we keep a buffer of all the steps that were taken using the interface abstraction,
    then we concatenate it all and package back as a single 'assistant' ChatCompletion response.

    The buffer needs to be wiped before each call to memgpt.agent.step()
    """

    def __init__(
        self,
        message_list=None,
        fancy=False,
        show_user_message=False,
        show_inner_thoughts=False,
        show_function_outputs=False,
        debug=False,
    ):
        self.message_list = message_list
        self.fancy = fancy  # set to false to disable colored outputs + emoji prefixes
        self.show_user_message = show_user_message
        self.show_inner_thoughts = show_inner_thoughts
        self.show_function_outputs = show_function_outputs
        self.debug = debug

    def reset_message_list(self):
        """Clears the buffer. Call before every agent.step() when using MemGPT+AutoGen"""
        self.message_list = []

    def internal_monologue(self, msg):
        # ANSI escape code for italic is '\x1B[3m'
        if self.debug:
            print(f"inner thoughts :: {msg}")
        if not self.show_inner_thoughts:
            return
        message = f"\x1B[3m{Fore.LIGHTBLACK_EX}💭 {msg}{Style.RESET_ALL}" if self.fancy else f"[inner thoughts] {msg}"
        self.message_list.append(message)

    def assistant_message(self, msg):
        if self.debug:
            print(f"assistant :: {msg}")
        message = f"{Fore.YELLOW}{Style.BRIGHT}🤖 {Fore.YELLOW}{msg}{Style.RESET_ALL}" if self.fancy else msg
        self.message_list.append(message)

    def memory_message(self, msg):
        if self.debug:
            print(f"memory :: {msg}")
        message = f"{Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}🧠 {Fore.LIGHTMAGENTA_EX}{msg}{Style.RESET_ALL}" if self.fancy else f"[memory] {msg}"
        self.message_list.append(message)

    def system_message(self, msg):
        if self.debug:
            print(f"system :: {msg}")
        message = f"{Fore.MAGENTA}{Style.BRIGHT}🖥️ [system] {Fore.MAGENTA}{msg}{Style.RESET_ALL}" if self.fancy else f"[system] {msg}"
        self.message_list.append(message)

    def user_message(self, msg, raw=False):
        if self.debug:
            print(f"user :: {msg}")
        if not self.show_user_message:
            return

        if isinstance(msg, str):
            if raw:
                message = f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg}{Style.RESET_ALL}" if self.fancy else f"[user] {msg}"
                self.message_list.append(message)
                return
            else:
                try:
                    msg_json = json.loads(msg)
                except:
                    print(f"Warning: failed to parse user message into json")
                    message = f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg}{Style.RESET_ALL}" if self.fancy else f"[user] {msg}"
                    self.message_list.append(message)
                    return

        if msg_json["type"] == "user_message":
            msg_json.pop("type")
            message = f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg_json}{Style.RESET_ALL}" if self.fancy else f"[user] {msg}"
        elif msg_json["type"] == "heartbeat":
            if True or DEBUG:
                msg_json.pop("type")
                message = (
                    f"{Fore.GREEN}{Style.BRIGHT}💓 {Fore.GREEN}{msg_json}{Style.RESET_ALL}" if self.fancy else f"[system heartbeat] {msg}"
                )
        elif msg_json["type"] == "system_message":
            msg_json.pop("type")
            message = f"{Fore.GREEN}{Style.BRIGHT}🖥️ {Fore.GREEN}{msg_json}{Style.RESET_ALL}" if self.fancy else f"[system] {msg}"
        else:
            message = f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg_json}{Style.RESET_ALL}" if self.fancy else f"[user] {msg}"

        self.message_list.append(message)

    def function_message(self, msg):
        if self.debug:
            print(f"function :: {msg}")
        if not self.show_function_outputs:
            return

        if isinstance(msg, dict):
            message = f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}"
            self.message_list.append(message)
            return

        if msg.startswith("Success: "):
            message = f"{Fore.RED}{Style.BRIGHT}⚡🟢 [function] {Fore.RED}{msg}{Style.RESET_ALL}" if self.fancy else f"[function - OK] {msg}"
        elif msg.startswith("Error: "):
            message = (
                f"{Fore.RED}{Style.BRIGHT}⚡🔴 [function] {Fore.RED}{msg}{Style.RESET_ALL}" if self.fancy else f"[function - error] {msg}"
            )
        elif msg.startswith("Running "):
            if DEBUG:
                message = f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}" if self.fancy else f"[function] {msg}"
            else:
                if "memory" in msg:
                    match = re.search(r"Running (\w+)\((.*)\)", msg)
                    if match:
                        function_name = match.group(1)
                        function_args = match.group(2)
                        message = (
                            f"{Fore.RED}{Style.BRIGHT}⚡🧠 [function] {Fore.RED}updating memory with {function_name}{Style.RESET_ALL}:"
                            if self.fancy
                            else f"[function] updating memory with {function_name}"
                        )
                        try:
                            msg_dict = eval(function_args)
                            if function_name == "archival_memory_search":
                                message = (
                                    f'{Fore.RED}\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                                    if self.fancy
                                    else f'[function] query: {msg_dict["query"]}, page: {msg_dict["page"]}'
                                )
                            else:
                                message = (
                                    f'{Fore.RED}{Style.BRIGHT}\t{Fore.RED} {msg_dict["old_content"]}\n\t{Fore.GREEN}→ {msg_dict["new_content"]}'
                                    if self.fancy
                                    else f'[old -> new] {msg_dict["old_content"]} -> {msg_dict["new_content"]}'
                                )
                        except Exception as e:
                            print(e)
                            message = msg_dict
                    else:
                        print(f"Warning: did not recognize function message")
                        message = (
                            f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}" if self.fancy else f"[function] {msg}"
                        )
                elif "send_message" in msg:
                    # ignore in debug mode
                    message = None
                else:
                    message = (
                        f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}" if self.fancy else f"[function] {msg}"
                    )
        else:
            try:
                msg_dict = json.loads(msg)
                if "status" in msg_dict and msg_dict["status"] == "OK":
                    message = (
                        f"{Fore.GREEN}{Style.BRIGHT}⚡ [function] {Fore.GREEN}{msg}{Style.RESET_ALL}" if self.fancy else f"[function] {msg}"
                    )
            except Exception:
                print(f"Warning: did not recognize function message {type(msg)} {msg}")
                message = f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}" if self.fancy else f"[function] {msg}"

        if message:
            self.message_list.append(message)
