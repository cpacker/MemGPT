import sys
import traceback

import questionary
import typer

from rich.console import Console
from memgpt.constants import FUNC_FAILED_HEARTBEAT_MESSAGE, REQ_HEARTBEAT_MESSAGE

console = Console()

from memgpt.interface import CLIInterface as interface  # for printing to terminal
from memgpt.config import MemGPTConfig
import memgpt.agent as agent
import memgpt.system as system
from memgpt.cli.cli import run
from memgpt.metadata import MetadataStore

# import benchmark

app = typer.Typer(pretty_exceptions_enable=False)
app.command(name="run")(run)


def clear_line():
    sys.stdout.write("\033[2K\033[G")
    sys.stdout.flush()


def run_agent_loop(memgpt_agent, config: MemGPTConfig, ms: MetadataStore):
    counter = 0
    user_input = None
    skip_next_user_input = False
    user_message = None

    multiline_input = False
    ms = MetadataStore(config)
    while True:
        if not skip_next_user_input and (counter > 0):
            # Ask for user input
            user_input = questionary.text(
                "Enter your message:",
                multiline=multiline_input,
                qmark=">",
            ).ask()
            clear_line()

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
                    # memgpt_agent.save()
                    agent.save_agent(memgpt_agent, ms)
                    break

                elif user_input.lower() == "//":
                    multiline_input = not multiline_input
                    continue

                else:
                    print(f"Unrecognized command: {user_input}")
                    continue

            else:
                # If message did not begin with command prefix, pass inputs to MemGPT
                # Handle user message and append to messages
                user_message = system.package_user_message(user_input)

        skip_next_user_input = False

        def process_agent_step(user_message):
            new_messages, heartbeat_request, function_failed, token_warning, tokens_accumulated = memgpt_agent.step(user_message)

            skip_next_user_input = False
            if token_warning:
                user_message = system.get_token_limit_warning()
                skip_next_user_input = True
            elif function_failed:
                user_message = system.get_heartbeat(FUNC_FAILED_HEARTBEAT_MESSAGE)
                skip_next_user_input = True
            elif heartbeat_request:
                user_message = system.get_heartbeat(REQ_HEARTBEAT_MESSAGE)
                skip_next_user_input = True

            return new_messages, user_message, skip_next_user_input

        while True:
            try:
                with console.status("[bold cyan]Thinking...") as _status:
                    new_messages, user_message, skip_next_user_input = process_agent_step(user_message)
                    break
            except KeyboardInterrupt:
                print("User interrupt occurred.")
                retry = questionary.confirm("Retry agent.step()?").ask()
                if not retry:
                    break
            except Exception as e:
                print("An exception occurred when running agent.step(): ")
                traceback.print_exc()
                retry = questionary.confirm("Retry agent.step()?").ask()
                if not retry:
                    break

        counter += 1

    print("Finished.")
