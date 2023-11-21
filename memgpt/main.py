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
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import (
    LocalStateManager,
    InMemoryStateManager,
    InMemoryStateManagerWithPreloadedArchivalMemory,
    InMemoryStateManagerWithFaiss,
)
from memgpt.cli.cli import run, attach, version
from memgpt.cli.cli_config import configure, list, add
from memgpt.cli.cli_load import app as load_app
from memgpt.config import Config, MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.agent import Agent
from memgpt.openai_tools import (
    configure_azure_support,
    check_azure_embeddings,
    get_set_azure_env_vars,
)
from memgpt.connectors.storage import StorageConnector

app = typer.Typer(pretty_exceptions_enable=False)
app.command(name="run")(run)
app.command(name="version")(version)
app.command(name="attach")(attach)
app.command(name="configure")(configure)
app.command(name="list")(list)
app.command(name="add")(add)
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


def save(memgpt_agent, cfg):
    filename = utils.get_local_time().replace(" ", "_").replace(":", "_")
    filename = f"{filename}.json"
    directory = os.path.join(MEMGPT_DIR, "saved_state")
    filename = os.path.join(directory, filename)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        memgpt_agent.save_to_json_file(filename)
        print(f"Saved checkpoint to: {filename}")
        cfg.agent_save_file = filename
    except Exception as e:
        print(f"Saving state to {filename} failed with: {e}")

    # save the persistence manager too
    filename = filename.replace(".json", ".persistence.pickle")
    try:
        memgpt_agent.persistence_manager.save(filename)
        print(f"Saved persistence manager to: {filename}")
        cfg.persistence_manager_save_file = filename
    except Exception as e:
        print(f"Saving persistence manager to {filename} failed with: {e}")
    cfg.write_config()


def load(memgpt_agent, filename):
    if filename is not None:
        if filename[-5:] != ".json":
            filename += ".json"
        try:
            memgpt_agent.load_from_json_file_inplace(filename)
            print(f"Loaded checkpoint {filename}")
        except Exception as e:
            print(f"Loading {filename} failed with: {e}")
    else:
        # Load the latest file
        save_path = os.path.join(constants.MEMGPT_DIR, "saved_state")
        print(f"/load warning: no checkpoint specified, loading most recent checkpoint from {save_path} instead")
        json_files = glob.glob(os.path.join(save_path, "*.json"))  # This will list all .json files in the current directory.

        # Check if there are any json files.
        if not json_files:
            print(f"/load error: no .json checkpoint files found")
            return
        else:
            # Sort files based on modified timestamp, with the latest file being the first.
            filename = max(json_files, key=os.path.getmtime)
            try:
                memgpt_agent.load_from_json_file_inplace(filename)
                print(f"Loaded checkpoint {filename}")
            except Exception as e:
                print(f"Loading {filename} failed with: {e}")

    # need to load persistence manager too
    filename = filename.replace(".json", ".persistence.pickle")
    try:
        memgpt_agent.persistence_manager = InMemoryStateManager.load(
            filename
        )  # TODO(fixme):for different types of persistence managers that require different load/save methods
        print(f"Loaded persistence manager from {filename}")
    except Exception as e:
        print(f"/load warning: loading persistence manager from {filename} failed with: {e}")


@app.callback(invoke_without_command=True)  # make default command
# @app.command("legacy-run")
def legacy_run(
    ctx: typer.Context,
    persona: str = typer.Option(None, help="Specify persona"),
    human: str = typer.Option(None, help="Specify human"),
    model: str = typer.Option(constants.DEFAULT_MEMGPT_MODEL, help="Specify the LLM model"),
    first: bool = typer.Option(False, "--first", help="Use --first to send the first message in the sequence"),
    strip_ui: bool = typer.Option(False, "--strip_ui", help="Remove all the bells and whistles in CLI output (helpful for testing)"),
    debug: bool = typer.Option(False, "--debug", help="Use --debug to enable debugging output"),
    no_verify: bool = typer.Option(False, "--no_verify", help="Bypass message verification"),
    archival_storage_faiss_path: str = typer.Option(
        "",
        "--archival_storage_faiss_path",
        help="Specify archival storage with FAISS index to load (a folder with a .index and .json describing documents to be loaded)",
    ),
    archival_storage_files: str = typer.Option(
        "",
        "--archival_storage_files",
        help="Specify files to pre-load into archival memory (glob pattern)",
    ),
    archival_storage_files_compute_embeddings: str = typer.Option(
        "",
        "--archival_storage_files_compute_embeddings",
        help="Specify files to pre-load into archival memory (glob pattern), and compute embeddings over them",
    ),
    archival_storage_sqldb: str = typer.Option(
        "",
        "--archival_storage_sqldb",
        help="Specify SQL database to pre-load into archival memory",
    ),
    use_azure_openai: bool = typer.Option(
        False,
        "--use_azure_openai",
        help="Use Azure OpenAI (requires additional environment variables)",
    ),  # TODO: just pass in?
):
    if ctx.invoked_subcommand is not None:
        return

    typer.secho(
        "Warning: Running legacy run command. You may need to `pip install pymemgpt[legacy] -U`. Run `memgpt run` instead.",
        fg=typer.colors.RED,
        bold=True,
    )
    if not questionary.confirm("Continue with legacy CLI?", default=False).ask():
        return

    main(
        persona,
        human,
        model,
        first,
        debug,
        no_verify,
        archival_storage_faiss_path,
        archival_storage_files,
        archival_storage_files_compute_embeddings,
        archival_storage_sqldb,
        use_azure_openai,
        strip_ui,
    )


def main(
    persona,
    human,
    model,
    first,
    debug,
    no_verify,
    archival_storage_faiss_path,
    archival_storage_files,
    archival_storage_files_compute_embeddings,
    archival_storage_sqldb,
    use_azure_openai,
    strip_ui,
):
    interface.STRIP_UI = strip_ui
    utils.DEBUG = debug
    logging.getLogger().setLevel(logging.CRITICAL)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Azure OpenAI support
    if use_azure_openai:
        configure_azure_support()
        check_azure_embeddings()
    else:
        azure_vars = get_set_azure_env_vars()
        if len(azure_vars) > 0:
            print(f"Error: Environment variables {', '.join([x[0] for x in azure_vars])} should not be set if --use_azure_openai is False")
            return

    if any(
        (
            persona,
            human,
            model != constants.DEFAULT_MEMGPT_MODEL,
            archival_storage_faiss_path,
            archival_storage_files,
            archival_storage_files_compute_embeddings,
            archival_storage_sqldb,
        )
    ):
        interface.important_message("⚙️ Using legacy command line arguments.")
        model = model
        if model is None:
            model = constants.DEFAULT_MEMGPT_MODEL
        memgpt_persona = persona
        if memgpt_persona is None:
            memgpt_persona = (
                personas.GPT35_DEFAULT if (model is not None and "gpt-3.5" in model) else personas.DEFAULT,
                None,  # represents the personas dir in pymemgpt package
            )
        else:
            try:
                personas.get_persona_text(memgpt_persona, Config.custom_personas_dir)
                memgpt_persona = (memgpt_persona, Config.custom_personas_dir)
            except FileNotFoundError:
                personas.get_persona_text(memgpt_persona)
                memgpt_persona = (memgpt_persona, None)

        human_persona = human
        if human_persona is None:
            human_persona = (humans.DEFAULT, None)
        else:
            try:
                humans.get_human_text(human_persona, Config.custom_humans_dir)
                human_persona = (human_persona, Config.custom_humans_dir)
            except FileNotFoundError:
                humans.get_human_text(human_persona)
                human_persona = (human_persona, None)

        print(persona, model, memgpt_persona)
        if archival_storage_files:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="folder",
                archival_storage_files=archival_storage_files,
                compute_embeddings=False,
            )
        elif archival_storage_faiss_path:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="folder",
                archival_storage_files=archival_storage_faiss_path,
                archival_storage_index=archival_storage_faiss_path,
                compute_embeddings=True,
            )
        elif archival_storage_files_compute_embeddings:
            print(model)
            print(memgpt_persona)
            print(human_persona)
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="folder",
                archival_storage_files=archival_storage_files_compute_embeddings,
                compute_embeddings=True,
            )
        elif archival_storage_sqldb:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="sql",
                archival_storage_files=archival_storage_sqldb,
                compute_embeddings=False,
            )
        else:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
            )
    else:
        cfg = Config.config_init()

    interface.important_message("Running... [exit by typing '/exit', list available commands with '/help']")
    if cfg.model != constants.DEFAULT_MEMGPT_MODEL:
        interface.warning_message(
            f"⛔️ Warning - you are running MemGPT with {cfg.model}, which is not officially supported (yet). Expect bugs!"
        )

    if cfg.index:
        persistence_manager = InMemoryStateManagerWithFaiss(cfg.index, cfg.archival_database)
    elif cfg.archival_storage_files:
        print(f"Preloaded {len(cfg.archival_database)} chunks into archival memory.")
        persistence_manager = InMemoryStateManagerWithPreloadedArchivalMemory(cfg.archival_database)
    else:
        persistence_manager = InMemoryStateManager()

    if archival_storage_files_compute_embeddings:
        interface.important_message(
            f"(legacy) To avoid computing embeddings next time, replace --archival_storage_files_compute_embeddings={archival_storage_files_compute_embeddings} with\n\t --archival_storage_faiss_path={cfg.archival_storage_index} (if your files haven't changed)."
        )

    # Moved defaults out of FLAGS so that we can dynamically select the default persona based on model
    chosen_human = cfg.human_persona
    chosen_persona = cfg.memgpt_persona

    memgpt_agent = presets.use_preset(
        presets.DEFAULT_PRESET,
        None,  # no agent config to provide
        cfg.model,
        personas.get_persona_text(*chosen_persona),
        humans.get_human_text(*chosen_human),
        interface,
        persistence_manager,
    )

    print_messages = interface.print_messages
    print_messages(memgpt_agent.messages)

    if cfg.load_type == "sql":  # TODO: move this into config.py in a clean manner
        if not os.path.exists(cfg.archival_storage_files):
            print(f"File {cfg.archival_storage_files} does not exist")
            return
        # Ingest data from file into archival storage
        else:
            print(f"Database found! Loading database into archival memory")
            data_list = utils.read_database_as_list(cfg.archival_storage_files)
            user_message = f"Your archival memory has been loaded with a SQL database called {data_list[0]}, which contains schema {data_list[1]}. Remember to refer to this first while answering any user questions!"
            for row in data_list:
                memgpt_agent.persistence_manager.archival_memory.insert(row)
            print(f"Database loaded into archival memory.")

    if cfg.agent_save_file:
        load_save_file = questionary.confirm(f"Load in saved agent '{cfg.agent_save_file}'?").ask()
        if load_save_file:
            load(memgpt_agent, cfg.agent_save_file)

    # run agent loop
    run_agent_loop(memgpt_agent, first, no_verify, cfg, strip_ui, legacy=True)


def run_agent_loop(memgpt_agent, first, no_verify=False, cfg=None, strip_ui=False, legacy=False):
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
                if legacy:
                    # legacy agent save functions (TODO: eventually remove)
                    if user_input.lower() == "/load" or user_input.lower().startswith("/load "):
                        command = user_input.strip().split()
                        filename = command[1] if len(command) > 1 else None
                        load(memgpt_agent=memgpt_agent, filename=filename)
                        continue
                    elif user_input.lower() == "/exit":
                        # autosave
                        save(memgpt_agent=memgpt_agent, cfg=cfg)
                        break

                    elif user_input.lower() == "/savechat":
                        filename = utils.get_local_time().replace(" ", "_").replace(":", "_")
                        filename = f"{filename}.pkl"
                        directory = os.path.join(MEMGPT_DIR, "saved_chats")
                        try:
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            with open(os.path.join(directory, filename), "wb") as f:
                                pickle.dump(memgpt_agent.messages, f)
                                print(f"Saved messages to: {filename}")
                        except Exception as e:
                            print(f"Saving chat to {filename} failed with: {e}")
                        continue

                    elif user_input.lower() == "/save":
                        save(memgpt_agent=memgpt_agent, cfg=cfg)
                        continue
                else:
                    # updated agent save functions
                    if user_input.lower() == "/exit":
                        memgpt_agent.save()
                        break
                    elif user_input.lower() == "/save" or user_input.lower() == "/savechat":
                        memgpt_agent.save()
                        continue

                if user_input.lower() == "/attach":
                    if legacy:
                        typer.secho("Error: /attach is not supported in legacy mode.", fg=typer.colors.RED, bold=True)
                        continue

                    # TODO: check if agent already has it
                    data_source_options = StorageConnector.list_loaded_data()
                    data_source = questionary.select("Select data source", choices=data_source_options).ask()

                    # attach new data
                    attach(memgpt_agent.config.name, data_source)

                    # update agent config
                    memgpt_agent.config.attach_data_source(data_source)

                    # reload agent with new data source
                    # TODO: maybe make this less ugly...
                    memgpt_agent.persistence_manager.archival_memory.storage = StorageConnector.get_storage_connector(
                        agent_config=memgpt_agent.config
                    )
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
                    amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
                    print(f"Popping last {amount} messages from stack")
                    for _ in range(min(amount, len(memgpt_agent.messages))):
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
                            memgpt_agent.messages[x].get("function_call").update({"arguments": json.dumps(args)})
                            break
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
