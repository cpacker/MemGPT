import shutil
import configparser
import uuid
import logging
import glob
import os
import sys
import pickle
import traceback
import json

import questionary
import typer

from rich.console import Console
from prettytable import PrettyTable

console = Console()

from memgpt.interface import CLIInterface as interface  # for printing to terminal
from memgpt.config import MemGPTConfig
import memgpt.agent as agent
import memgpt.system as system
import memgpt.constants as constants
import memgpt.errors as errors
from memgpt.cli.cli import run, attach, version, server, open_folder, quickstart, suppress_stdout
from memgpt.cli.cli_config import configure, list, add, delete
from memgpt.cli.cli_load import app as load_app
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.metadata import MetadataStore

app = typer.Typer(pretty_exceptions_enable=False)
app.command(name="run")(run)
app.command(name="version")(version)
app.command(name="attach")(attach)
app.command(name="configure")(configure)
app.command(name="list")(list)
app.command(name="add")(add)
app.command(name="delete")(delete)
app.command(name="server")(server)
app.command(name="folder")(open_folder)
app.command(name="quickstart")(quickstart)
# load data commands
app.add_typer(load_app, name="load")


def clear_line(strip_ui=False):
    if strip_ui:
        return
    if os.name == "nt":  # for windows
        console.print("\033[A\033[K", end="")
    else:  # for linux
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()


def run_agent_loop(memgpt_agent, config: MemGPTConfig, first, no_verify=False, cfg=None, strip_ui=False):
    counter = 0
    user_input = None
    skip_next_user_input = False
    user_message = None
    USER_GOES_FIRST = first

    if not USER_GOES_FIRST:
        console.input("[bold cyan]Hit enter to begin (will request first MemGPT message)[/bold cyan]")
        clear_line(strip_ui)
        print()

    multiline_input = False
    ms = MetadataStore(config)
    while True:
        if not skip_next_user_input and (counter > 0 or USER_GOES_FIRST):
            # Ask for user input
            user_input = questionary.text(
                "Enter your message:",
                multiline=multiline_input,
                qmark=">",
            ).ask()
            clear_line(strip_ui)

            # Gracefully exit on Ctrl-C/D
            if user_input is None:
                user_input = "/exit"

            user_input = user_input.rstrip()

            if user_input.startswith("!"):
                print(f"Commands for CLI begin with '/' not '!'")
                continue

            if user_input == "":
                # no empty messages allowed
                print("Empty input received. Try again!")
                continue

            # Handle CLI commands
            # Commands to not get passed as input to MemGPT
            if user_input.startswith("/"):
                # updated agent save functions
                if user_input.lower() == "/exit":
                    memgpt_agent.save()
                    break
                elif user_input.lower() == "/save" or user_input.lower() == "/savechat":
                    memgpt_agent.save()
                    continue
                elif user_input.lower() == "/attach":
                    # TODO: check if agent already has it

                    data_source_options = ms.list_sources(user_id=memgpt_agent.agent_state.user_id)
                    data_source_options = [s.name for s in data_source_options]
                    if len(data_source_options) == 0:
                        typer.secho(
                            'No sources available. You must load a souce with "memgpt load ..." before running /attach.',
                            fg=typer.colors.RED,
                            bold=True,
                        )
                        continue
                    data_source = questionary.select("Select data source", choices=data_source_options).ask()

                    # attach new data
                    attach(memgpt_agent.config.name, data_source)

                    continue

                elif user_input.lower() == "/dump" or user_input.lower().startswith("/dump "):
                    # Check if there's an additional argument that's an integer
                    command = user_input.strip().split()
                    amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 0
                    if amount == 0:
                        interface.print_messages(memgpt_agent.messages, dump=True)
                    else:
                        interface.print_messages(memgpt_agent.messages[-min(amount, len(memgpt_agent.messages)) :], dump=True)
                    continue

                elif user_input.lower() == "/dumpraw":
                    interface.print_messages_raw(memgpt_agent.messages)
                    continue

                elif user_input.lower() == "/memory":
                    print(f"\nDumping memory contents:\n")
                    print(f"{str(memgpt_agent.memory)}")
                    print(f"{str(memgpt_agent.persistence_manager.archival_memory)}")
                    print(f"{str(memgpt_agent.persistence_manager.recall_memory)}")
                    continue

                elif user_input.lower() == "/model":
                    if memgpt_agent.model == "gpt-4":
                        memgpt_agent.model = "gpt-3.5-turbo-16k"
                    elif memgpt_agent.model == "gpt-3.5-turbo-16k":
                        memgpt_agent.model = "gpt-4"
                    print(f"Updated model to:\n{str(memgpt_agent.model)}")
                    continue

                elif user_input.lower() == "/pop" or user_input.lower().startswith("/pop "):
                    # Check if there's an additional argument that's an integer
                    command = user_input.strip().split()
                    pop_amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
                    n_messages = len(memgpt_agent.messages)
                    MIN_MESSAGES = 2
                    if n_messages <= MIN_MESSAGES:
                        print(f"Agent only has {n_messages} messages in stack, none left to pop")
                    elif n_messages - pop_amount < MIN_MESSAGES:
                        print(f"Agent only has {n_messages} messages in stack, cannot pop more than {n_messages - MIN_MESSAGES}")
                    else:
                        print(f"Popping last {pop_amount} messages from stack")
                        for _ in range(min(pop_amount, len(memgpt_agent.messages))):
                            memgpt_agent.messages.pop()
                    continue

                elif user_input.lower() == "/retry":
                    # TODO this needs to also modify the persistence manager
                    print(f"Retrying for another answer")
                    while len(memgpt_agent.messages) > 0:
                        if memgpt_agent.messages[-1].get("role") == "user":
                            # we want to pop up to the last user message and send it again
                            user_message = memgpt_agent.messages[-1].get("content")
                            memgpt_agent.messages.pop()
                            break
                        memgpt_agent.messages.pop()

                elif user_input.lower() == "/rethink" or user_input.lower().startswith("/rethink "):
                    # TODO this needs to also modify the persistence manager
                    if len(user_input) < len("/rethink "):
                        print("Missing text after the command")
                        continue
                    for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                        if memgpt_agent.messages[x].get("role") == "assistant":
                            text = user_input[len("/rethink ") :].strip()
                            memgpt_agent.messages[x].update({"content": text})
                            break
                    continue

                elif user_input.lower() == "/rewrite" or user_input.lower().startswith("/rewrite "):
                    # TODO this needs to also modify the persistence manager
                    if len(user_input) < len("/rewrite "):
                        print("Missing text after the command")
                        continue
                    for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                        if memgpt_agent.messages[x].get("role") == "assistant":
                            text = user_input[len("/rewrite ") :].strip()
                            args = json.loads(memgpt_agent.messages[x].get("function_call").get("arguments"))
                            args["message"] = text
                            memgpt_agent.messages[x].get("function_call").update({"arguments": json.dumps(args, ensure_ascii=False)})
                            break
                    continue

                elif user_input.lower() == "/summarize":
                    try:
                        memgpt_agent.summarize_messages_inplace()
                        typer.secho(
                            f"/summarize succeeded",
                            fg=typer.colors.GREEN,
                            bold=True,
                        )
                    except errors.LLMError as e:
                        typer.secho(
                            f"/summarize failed:\n{e}",
                            fg=typer.colors.RED,
                            bold=True,
                        )
                    continue

                # No skip options
                elif user_input.lower() == "/wipe":
                    memgpt_agent = agent.Agent(interface)
                    user_message = None

                elif user_input.lower() == "/heartbeat":
                    user_message = system.get_heartbeat()

                elif user_input.lower() == "/memorywarning":
                    user_message = system.get_token_limit_warning()

                elif user_input.lower() == "//":
                    multiline_input = not multiline_input
                    continue

                elif user_input.lower() == "/" or user_input.lower() == "/help":
                    questionary.print("CLI commands", "bold")
                    for cmd, desc in USER_COMMANDS:
                        questionary.print(cmd, "bold")
                        questionary.print(f" {desc}")
                    continue

                else:
                    print(f"Unrecognized command: {user_input}")
                    continue

            else:
                # If message did not begin with command prefix, pass inputs to MemGPT
                # Handle user message and append to messages
                user_message = system.package_user_message(user_input)

        skip_next_user_input = False

        def process_agent_step(user_message, no_verify):
            new_messages, heartbeat_request, function_failed, token_warning = memgpt_agent.step(
                user_message, first_message=False, skip_verify=no_verify
            )

            skip_next_user_input = False
            if token_warning:
                user_message = system.get_token_limit_warning()
                skip_next_user_input = True
            elif function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                skip_next_user_input = True
            elif heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                skip_next_user_input = True

            return new_messages, user_message, skip_next_user_input

        while True:
            try:
                if strip_ui:
                    new_messages, user_message, skip_next_user_input = process_agent_step(user_message, no_verify)
                    break
                else:
                    with console.status("[bold cyan]Thinking...") as status:
                        new_messages, user_message, skip_next_user_input = process_agent_step(user_message, no_verify)
                        break
            except KeyboardInterrupt:
                print("User interrupt occured.")
                retry = questionary.confirm("Retry agent.step()?").ask()
                if not retry:
                    break
            except Exception as e:
                print("An exception ocurred when running agent.step(): ")
                traceback.print_exc()
                retry = questionary.confirm("Retry agent.step()?").ask()
                if not retry:
                    break

        counter += 1

    print("Finished.")


USER_COMMANDS = [
    ("//", "toggle multiline input mode"),
    ("/exit", "exit the CLI"),
    ("/save", "save a checkpoint of the current agent/conversation state"),
    ("/load", "load a saved checkpoint"),
    ("/dump <count>", "view the last <count> messages (all if <count> is omitted)"),
    ("/memory", "print the current contents of agent memory"),
    ("/pop <count>", "undo <count> messages in the conversation (default is 3)"),
    ("/retry", "pops the last answer and tries to get another one"),
    ("/rethink <text>", "changes the inner thoughts of the last agent message"),
    ("/rewrite <text>", "changes the reply of the last agent message"),
    ("/heartbeat", "send a heartbeat system message to the agent"),
    ("/memorywarning", "send a memory warning system message to the agent"),
    ("/attach", "attach data source to agent"),
]
