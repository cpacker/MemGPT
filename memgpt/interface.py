import json
import re

from colorama import Fore, Style, init

from memgpt.utils import printd

init(autoreset=True)

# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal


def important_message(msg):
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{msg}{Style.RESET_ALL}")


def warning_message(msg):
    print(f"{Fore.RED}{Style.BRIGHT}{msg}{Style.RESET_ALL}")


async def internal_monologue(msg):
    # ANSI escape code for italic is '\x1B[3m'
    print(f"\x1B[3m{Fore.LIGHTBLACK_EX}💭 {msg}{Style.RESET_ALL}")


async def assistant_message(msg):
    print(f"{Fore.YELLOW}{Style.BRIGHT}🤖 {Fore.YELLOW}{msg}{Style.RESET_ALL}")


async def memory_message(msg):
    print(f"{Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}🧠 {Fore.LIGHTMAGENTA_EX}{msg}{Style.RESET_ALL}")


async def system_message(msg):
    printd(f"{Fore.MAGENTA}{Style.BRIGHT}🖥️ [system] {Fore.MAGENTA}{msg}{Style.RESET_ALL}")


async def user_message(msg, raw=False):
    if isinstance(msg, str):
        if raw:
            printd(f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg}{Style.RESET_ALL}")
            return
        else:
            try:
                msg_json = json.loads(msg)
            except:
                printd(f"Warning: failed to parse user message into json")
                printd(f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg}{Style.RESET_ALL}")
                return

    if msg_json["type"] == "user_message":
        msg_json.pop("type")
        printd(f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg_json}{Style.RESET_ALL}")
    elif msg_json["type"] == "heartbeat":
        if DEBUG:
            msg_json.pop("type")
            printd(f"{Fore.GREEN}{Style.BRIGHT}💓 {Fore.GREEN}{msg_json}{Style.RESET_ALL}")
    elif msg_json["type"] == "system_message":
        msg_json.pop("type")
        printd(f"{Fore.GREEN}{Style.BRIGHT}🖥️ {Fore.GREEN}{msg_json}{Style.RESET_ALL}")
    else:
        printd(f"{Fore.GREEN}{Style.BRIGHT}🧑 {Fore.GREEN}{msg_json}{Style.RESET_ALL}")


async def function_message(msg):
    if isinstance(msg, dict):
        printd(f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}")
        return

    if msg.startswith("Success: "):
        printd(f"{Fore.RED}{Style.BRIGHT}⚡🟢 [function] {Fore.RED}{msg}{Style.RESET_ALL}")
    elif msg.startswith("Error: "):
        printd(f"{Fore.RED}{Style.BRIGHT}⚡🔴 [function] {Fore.RED}{msg}{Style.RESET_ALL}")
    elif msg.startswith("Running "):
        if DEBUG:
            printd(f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}")
        else:
            if "memory" in msg:
                match = re.search(r"Running (\w+)\((.*)\)", msg)
                if match:
                    function_name = match.group(1)
                    function_args = match.group(2)
                    print(f"{Fore.RED}{Style.BRIGHT}⚡🧠 [function] {Fore.RED}updating memory with {function_name}{Style.RESET_ALL}:")
                    try:
                        msg_dict = eval(function_args)
                        if function_name == "archival_memory_search":
                            print(f'{Fore.RED}\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}')
                        else:
                            print(
                                f'{Fore.RED}{Style.BRIGHT}\t{Fore.RED} {msg_dict["old_content"]}\n\t{Fore.GREEN}→ {msg_dict["new_content"]}'
                            )
                    except Exception as e:
                        printd(e)
                        printd(msg_dict)
                        pass
                else:
                    printd(f"Warning: did not recognize function message")
                    printd(f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}")
            elif "send_message" in msg:
                # ignore in debug mode
                pass
            else:
                printd(f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}")
    else:
        try:
            msg_dict = json.loads(msg)
            if "status" in msg_dict and msg_dict["status"] == "OK":
                printd(f"{Fore.GREEN}{Style.BRIGHT}⚡ [function] {Fore.GREEN}{msg}{Style.RESET_ALL}")
        except Exception:
            printd(f"Warning: did not recognize function message {type(msg)} {msg}")
            printd(f"{Fore.RED}{Style.BRIGHT}⚡ [function] {Fore.RED}{msg}{Style.RESET_ALL}")


async def print_messages(message_sequence):
    for msg in message_sequence:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            await system_message(content)
        elif role == "assistant":
            # Differentiate between internal monologue, function calls, and messages
            if msg.get("function_call"):
                if content is not None:
                    await internal_monologue(content)
                await function_message(msg["function_call"])
                # assistant_message(content)
            else:
                await internal_monologue(content)
        elif role == "user":
            await user_message(content)
        elif role == "function":
            await function_message(content)
        else:
            print(f"Unknown role: {content}")


async def print_messages_simple(message_sequence):
    for msg in message_sequence:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            await system_message(content)
        elif role == "assistant":
            await assistant_message(content)
        elif role == "user":
            await user_message(content, raw=True)
        else:
            print(f"Unknown role: {content}")


async def print_messages_raw(message_sequence):
    for msg in message_sequence:
        print(msg)
