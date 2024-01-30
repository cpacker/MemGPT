import uuid
import json
import requests
import sys
import shutil
import io
import logging
from pathlib import Path
import os
import subprocess
from enum import Enum
from typing import Annotated, Optional

import typer
import questionary
from llama_index import set_global_service_context
from llama_index import ServiceContext

from memgpt.log import logger
from memgpt.interface import CLIInterface as interface  # for printing to terminal
from memgpt.cli.cli_config import configure
import memgpt.presets.presets as presets
import memgpt.utils as utils
from memgpt.utils import printd, open_folder_in_explorer, suppress_stdout
from memgpt.config import MemGPTConfig
from memgpt.credentials import MemGPTCredentials
from memgpt.constants import MEMGPT_DIR, CLI_WARNING_PREFIX, JSON_ENSURE_ASCII
from memgpt.agent import Agent
from memgpt.embeddings import embedding_model
from memgpt.server.constants import WS_DEFAULT_PORT, REST_DEFAULT_PORT
from memgpt.data_types import AgentState, LLMConfig, EmbeddingConfig, User, Passage
from memgpt.metadata import MetadataStore, save_agent
from memgpt.migrate import migrate_all_agents, migrate_all_sources


def migrate(
    debug: Annotated[bool, typer.Option(help="Print extra tracebacks for failed migrations")] = False,
):
    """Migrate old agents (pre 0.2.12) to the new database system"""
    migrate_all_agents(debug=debug)
    migrate_all_sources(debug=debug)


class QuickstartChoice(Enum):
    openai = "openai"
    # azure = "azure"
    memgpt_hosted = "memgpt"


def str_to_quickstart_choice(choice_str: str) -> QuickstartChoice:
    try:
        return QuickstartChoice[choice_str]
    except KeyError:
        valid_options = [choice.name for choice in QuickstartChoice]
        raise ValueError(f"{choice_str} is not a valid QuickstartChoice. Valid options are: {valid_options}")


def set_config_with_dict(new_config: dict) -> bool:
    """Set the base config using a dict"""
    from memgpt.utils import printd

    old_config = MemGPTConfig.load()
    modified = False
    for k, v in vars(old_config).items():
        if k in new_config:
            if v != new_config[k]:
                printd(f"Replacing config {k}: {v} -> {new_config[k]}")
                modified = True
                # old_config[k] = new_config[k]
                setattr(old_config, k, new_config[k])  # Set the new value using dot notation
            else:
                printd(f"Skipping new config {k}: {v} == {new_config[k]}")

    # update embedding config
    for k, v in vars(old_config.default_embedding_config).items():
        if k in new_config:
            if v != new_config[k]:
                printd(f"Replacing config {k}: {v} -> {new_config[k]}")
                modified = True
                # old_config[k] = new_config[k]
                setattr(old_config.default_embedding_config, k, new_config[k])
        else:
            printd(f"Skipping new config {k}: {v} == {new_config[k]}")

    # update llm config
    for k, v in vars(old_config.default_llm_config).items():
        if k in new_config:
            if v != new_config[k]:
                printd(f"Replacing config {k}: {v} -> {new_config[k]}")
                modified = True
                # old_config[k] = new_config[k]
                setattr(old_config.default_llm_config, k, new_config[k])
        else:
            printd(f"Skipping new config {k}: {v} == {new_config[k]}")

    if modified:
        printd(f"Saving new config file.")
        old_config.save()
        typer.secho(f"📖 MemGPT configuration file updated!", fg=typer.colors.GREEN)
        typer.secho(
            "\n".join(
                [
                    f"🧠 model\t-> {old_config.default_llm_config.model}",
                    f"🖥️  endpoint\t-> {old_config.default_llm_config.model_endpoint}",
                ]
            ),
            fg=typer.colors.GREEN,
        )
        return True
    else:
        typer.secho(f"📖 MemGPT configuration file unchanged.", fg=typer.colors.WHITE)
        typer.secho(
            "\n".join(
                [
                    f"🧠 model\t-> {old_config.default_llm_config.model}",
                    f"🖥️  endpoint\t-> {old_config.default_llm_config.model_endpoint}",
                ]
            ),
            fg=typer.colors.WHITE,
        )
        return False


def quickstart(
    backend: Annotated[QuickstartChoice, typer.Option(help="Quickstart setup backend")] = "memgpt",
    latest: Annotated[bool, typer.Option(help="Use --latest to pull the latest config from online")] = False,
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
    terminal: bool = True,
):
    """Set the base config file with a single command"""

    # setup logger
    utils.DEBUG = debug
    logging.getLogger().setLevel(logging.CRITICAL)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # make sure everything is set up properly
    MemGPTConfig.create_config_dir()
    credentials = MemGPTCredentials.load()

    config_was_modified = False
    if backend == QuickstartChoice.memgpt_hosted:
        # if latest, try to pull the config from the repo
        # fallback to using local
        if latest:
            # Download the latest memgpt hosted config
            url = "https://raw.githubusercontent.com/cpacker/MemGPT/main/memgpt/configs/memgpt_hosted.json"
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response content as JSON
                config = response.json()
                # Output a success message and the first few items in the dictionary as a sample
                printd("JSON config file downloaded successfully.")
                config_was_modified = set_config_with_dict(config)
            else:
                typer.secho(f"Failed to download config from {url}. Status code: {response.status_code}", fg=typer.colors.RED)

                # Load the file from the relative path
                script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
                backup_config_path = os.path.join(script_dir, "..", "configs", "memgpt_hosted.json")
                try:
                    with open(backup_config_path, "r", encoding="utf-8") as file:
                        backup_config = json.load(file)
                    printd("Loaded backup config file successfully.")
                    config_was_modified = set_config_with_dict(backup_config)
                except FileNotFoundError:
                    typer.secho(f"Backup config file not found at {backup_config_path}", fg=typer.colors.RED)
                    return
        else:
            # Load the file from the relative path
            script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
            backup_config_path = os.path.join(script_dir, "..", "configs", "memgpt_hosted.json")
            try:
                with open(backup_config_path, "r", encoding="utf-8") as file:
                    backup_config = json.load(file)
                printd("Loaded config file successfully.")
                config_was_modified = set_config_with_dict(backup_config)
            except FileNotFoundError:
                typer.secho(f"Config file not found at {backup_config_path}", fg=typer.colors.RED)
                return

    elif backend == QuickstartChoice.openai:
        # Make sure we have an API key
        api_key = os.getenv("OPENAI_API_KEY")
        while api_key is None or len(api_key) == 0:
            # Ask for API key as input
            api_key = questionary.password("Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):").ask()
        credentials.openai_key = api_key
        credentials.save()

        # if latest, try to pull the config from the repo
        # fallback to using local
        if latest:
            url = "https://raw.githubusercontent.com/cpacker/MemGPT/main/memgpt/configs/openai.json"
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response content as JSON
                config = response.json()
                # Output a success message and the first few items in the dictionary as a sample
                print("JSON config file downloaded successfully.")
                config_was_modified = set_config_with_dict(config)
            else:
                typer.secho(f"Failed to download config from {url}. Status code: {response.status_code}", fg=typer.colors.RED)

                # Load the file from the relative path
                script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
                backup_config_path = os.path.join(script_dir, "..", "configs", "openai.json")
                try:
                    with open(backup_config_path, "r", encoding="utf-8") as file:
                        backup_config = json.load(file)
                    printd("Loaded backup config file successfully.")
                    config_was_modified = set_config_with_dict(backup_config)
                except FileNotFoundError:
                    typer.secho(f"Backup config file not found at {backup_config_path}", fg=typer.colors.RED)
                    return
        else:
            # Load the file from the relative path
            script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
            backup_config_path = os.path.join(script_dir, "..", "configs", "openai.json")
            try:
                with open(backup_config_path, "r", encoding="utf-8") as file:
                    backup_config = json.load(file)
                printd("Loaded config file successfully.")
                config_was_modified = set_config_with_dict(backup_config)
            except FileNotFoundError:
                typer.secho(f"Config file not found at {backup_config_path}", fg=typer.colors.RED)
                return

    else:
        raise NotImplementedError(backend)

    # 'terminal' = quickstart was run alone, in which case we should guide the user on the next command
    if terminal:
        if config_was_modified:
            typer.secho('⚡ Run "memgpt run" to create an agent with the new config.', fg=typer.colors.YELLOW)
        else:
            typer.secho('⚡ Run "memgpt run" to create an agent.', fg=typer.colors.YELLOW)


def open_folder():
    """Open a folder viewer of the MemGPT home directory"""
    try:
        print(f"Opening home folder: {MEMGPT_DIR}")
        open_folder_in_explorer(MEMGPT_DIR)
    except Exception as e:
        print(f"Failed to open folder with system viewer, error:\n{e}")


class ServerChoice(Enum):
    rest_api = "rest"
    ws_api = "websocket"


def create_default_user_or_exit(config: MemGPTConfig, ms: MetadataStore):
    user_id = uuid.UUID(config.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        ms.create_user(User(id=user_id))
        user = ms.get_user(user_id=user_id)
        if user is None:
            typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
            sys.exit(1)
        else:
            return user
    else:
        return user


def server(
    type: Annotated[ServerChoice, typer.Option(help="Server to run")] = "rest",
    port: Annotated[Optional[int], typer.Option(help="Port to run the server on")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host to run the server on (default to localhost)")] = None,
    debug: Annotated[bool, typer.Option(help="Turn debugging output on")] = True,
):
    """Launch a MemGPT server process"""

    if debug:
        from memgpt.server.server import logger as server_logger

        # Set the logging level
        server_logger.setLevel(logging.DEBUG)
        # Create a StreamHandler
        stream_handler = logging.StreamHandler()
        # Set the formatter (optional)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        # Add the handler to the logger
        server_logger.addHandler(stream_handler)

    if type == ServerChoice.rest_api:
        import uvicorn
        from memgpt.server.rest_api.server import app

        if MemGPTConfig.exists():
            config = MemGPTConfig.load()
            ms = MetadataStore(config)
            create_default_user_or_exit(config, ms)
        else:
            typer.secho(f"No configuration exists. Run memgpt configure before starting the server.", fg=typer.colors.RED)
            sys.exit(1)

        try:
            # Start the subprocess in a new session
            uvicorn.run(app, host=host or "localhost", port=port or REST_DEFAULT_PORT)

        except KeyboardInterrupt:
            # Handle CTRL-C
            typer.secho("Terminating the server...")
            sys.exit(0)

    elif type == ServerChoice.ws_api:
        if port is None:
            port = WS_DEFAULT_PORT

        # Change to the desired directory
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent

        server_directory = os.path.join(script_dir.parent, "server", "ws_api")
        command = f"python server.py {port}"

        # Run the command
        typer.secho(f"Running WS (websockets) server: {command} (inside {server_directory})")

        process = None
        try:
            # Start the subprocess in a new session
            process = subprocess.Popen(command, shell=True, start_new_session=True, cwd=server_directory)
            process.wait()
        except KeyboardInterrupt:
            # Handle CTRL-C
            if process is not None:
                typer.secho("Terminating the server...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    typer.secho("Server terminated with kill()")
            sys.exit(0)


def run(
    persona: Annotated[Optional[str], typer.Option(help="Specify persona")] = None,
    agent: Annotated[Optional[str], typer.Option(help="Specify agent save file")] = None,
    human: Annotated[Optional[str], typer.Option(help="Specify human")] = None,
    preset: Annotated[Optional[str], typer.Option(help="Specify preset")] = None,
    # model flags
    model: Annotated[Optional[str], typer.Option(help="Specify the LLM model")] = None,
    model_wrapper: Annotated[Optional[str], typer.Option(help="Specify the LLM model wrapper")] = None,
    model_endpoint: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint")] = None,
    model_endpoint_type: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint type")] = None,
    context_window: Annotated[
        Optional[int], typer.Option(help="The context window of the LLM you are using (e.g. 8k for most Mistral 7B variants)")
    ] = None,
    # other
    first: Annotated[bool, typer.Option(help="Use --first to send the first message in the sequence")] = False,
    strip_ui: Annotated[bool, typer.Option(help="Remove all the bells and whistles in CLI output (helpful for testing)")] = False,
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
    no_verify: Annotated[bool, typer.Option(help="Bypass message verification")] = False,
    yes: Annotated[bool, typer.Option("-y", help="Skip confirmation prompt and use defaults")] = False,
):
    """Start chatting with an MemGPT agent

    Example usage: `memgpt run --agent myagent --data-source mydata --persona mypersona --human myhuman --model gpt-3.5-turbo`

    :param persona: Specify persona
    :param agent: Specify agent name (will load existing state if the agent exists, or create a new one with that name)
    :param human: Specify human
    :param model: Specify the LLM model

    """

    # setup logger
    # TODO: remove Utils Debug after global logging is complete.
    utils.DEBUG = debug
    # TODO: add logging command line options for runtime log level

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    from memgpt.migrate import config_is_compatible, wipe_config_and_reconfigure, VERSION_CUTOFF

    if not config_is_compatible(allow_empty=True):
        typer.secho(f"\nYour current config file is incompatible with MemGPT versions later than {VERSION_CUTOFF}\n", fg=typer.colors.RED)
        choices = [
            "Run the full config setup (recommended)",
            "Create a new config using defaults",
            "Cancel",
        ]
        selection = questionary.select(
            f"To use MemGPT, you must either downgrade your MemGPT version (<= {VERSION_CUTOFF}), or regenerate your config. Would you like to proceed?",
            choices=choices,
            default=choices[0],
        ).ask()
        if selection == choices[0]:
            try:
                wipe_config_and_reconfigure()
            except Exception as e:
                typer.secho(f"Fresh config generation failed - error:\n{e}", fg=typer.colors.RED)
                raise
        elif selection == choices[1]:
            try:
                # Don't create a config, so that the next block of code asking about quickstart is run
                wipe_config_and_reconfigure(run_configure=False, create_config=False)
            except Exception as e:
                typer.secho(f"Fresh config generation failed - error:\n{e}", fg=typer.colors.RED)
                raise
        else:
            typer.secho("MemGPT config regeneration cancelled", fg=typer.colors.RED)
            raise KeyboardInterrupt()

        typer.secho("Note: if you would like to migrate old agents to the new release, please run `memgpt migrate`!", fg=typer.colors.GREEN)

    if not MemGPTConfig.exists():
        # if no config, ask about quickstart
        # do you want to do:
        # - openai (run quickstart)
        # - memgpt hosted (run quickstart)
        # - other (run configure)
        if yes:
            # if user is passing '-y' to bypass all inputs, use memgpt hosted
            # since it can't fail out if you don't have an API key
            quickstart(backend=QuickstartChoice.memgpt_hosted)
            config = MemGPTConfig()

        else:
            config_choices = {
                "memgpt": "Use the free MemGPT endpoints",
                "openai": "Use OpenAI (requires an OpenAI API key)",
                "other": "Other (OpenAI Azure, custom LLM endpoint, etc)",
            }
            print()
            config_selection = questionary.select(
                "How would you like to set up MemGPT?",
                choices=list(config_choices.values()),
                default=config_choices["memgpt"],
            ).ask()

            if config_selection == config_choices["memgpt"]:
                print()
                quickstart(backend=QuickstartChoice.memgpt_hosted, debug=debug, terminal=False, latest=False)
            elif config_selection == config_choices["openai"]:
                print()
                quickstart(backend=QuickstartChoice.openai, debug=debug, terminal=False, latest=False)
            elif config_selection == config_choices["other"]:
                configure()
            else:
                raise ValueError(config_selection)

            config = MemGPTConfig.load()

    else:  # load config
        config = MemGPTConfig.load()

        # force re-configuration is config is from old version
        if config.memgpt_version is None:  # TODO: eventually add checks for older versions, if config changes again
            typer.secho("MemGPT has been updated to a newer version, so re-running configuration.", fg=typer.colors.YELLOW)
            configure()
            config = MemGPTConfig.load()

    # read user id from config
    ms = MetadataStore(config)
    user = create_default_user_or_exit(config, ms)

    # determine agent to use, if not provided
    if not yes and not agent:
        agents = ms.list_agents(user_id=user.id)
        agents = [a.name for a in agents]

        if len(agents) > 0 and not any([persona, human, model]):
            print()
            select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
            if select_agent is None:
                raise KeyboardInterrupt
            if select_agent:
                agent = questionary.select("Select agent:", choices=agents).ask()

    # create agent config
    agent_state = ms.get_agent(agent_name=agent, user_id=user.id) if agent else None
    if agent and agent_state:  # use existing agent
        typer.secho(f"\n🔁 Using existing agent {agent}", fg=typer.colors.GREEN)
        # agent_config = AgentConfig.load(agent)
        # agent_state = ms.get_agent(agent_name=agent, user_id=user_id)
        printd("Loading agent state:", agent_state.id)
        printd("Agent state:", agent_state.state)
        # printd("State path:", agent_config.save_state_dir())
        # printd("Persistent manager path:", agent_config.save_persistence_manager_dir())
        # printd("Index path:", agent_config.save_agent_index_dir())
        # persistence_manager = LocalStateManager(agent_config).load() # TODO: implement load
        # TODO: load prior agent state
        if persona and persona != agent_state.persona:
            typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing persona {agent_state.persona} with {persona}", fg=typer.colors.YELLOW)
            agent_state.persona = persona
            # raise ValueError(f"Cannot override {agent_state.name} existing persona {agent_state.persona} with {persona}")
        if human and human != agent_state.human:
            typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing human {agent_state.human} with {human}", fg=typer.colors.YELLOW)
            agent_state.human = human
            # raise ValueError(f"Cannot override {agent_config.name} existing human {agent_config.human} with {human}")

        # Allow overriding model specifics (model, model wrapper, model endpoint IP + type, context_window)
        if model and model != agent_state.llm_config.model:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model {agent_state.llm_config.model} with {model}", fg=typer.colors.YELLOW
            )
            agent_state.llm_config.model = model
        if context_window is not None and int(context_window) != agent_state.llm_config.context_window:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing context window {agent_state.llm_config.context_window} with {context_window}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.context_window = context_window
        if model_wrapper and model_wrapper != agent_state.llm_config.model_wrapper:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model wrapper {agent_state.llm_config.model_wrapper} with {model_wrapper}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_wrapper = model_wrapper
        if model_endpoint and model_endpoint != agent_state.llm_config.model_endpoint:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint {agent_state.llm_config.model_endpoint} with {model_endpoint}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_endpoint = model_endpoint
        if model_endpoint_type and model_endpoint_type != agent_state.llm_config.model_endpoint_type:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint type {agent_state.llm_config.model_endpoint_type} with {model_endpoint_type}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_endpoint_type = model_endpoint_type

        # Update the agent with any overrides
        ms.update_agent(agent_state)

        # create agent
        memgpt_agent = Agent(agent_state, interface=interface())

    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)

        if agent is None:
            # determine agent name
            # agent_count = len(ms.list_agents(user_id=user.id))
            # agent = f"agent_{agent_count}"
            agent = utils.create_random_username()

        llm_config = config.default_llm_config
        embedding_config = config.default_embedding_config  # TODO allow overriding embedding params via CLI run

        # Allow overriding model specifics (model, model wrapper, model endpoint IP + type, context_window)
        if model and model != llm_config.model:
            typer.secho(f"{CLI_WARNING_PREFIX}Overriding default model {llm_config.model} with {model}", fg=typer.colors.YELLOW)
            llm_config.model = model
        if context_window is not None and int(context_window) != llm_config.context_window:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding default context window {llm_config.context_window} with {context_window}",
                fg=typer.colors.YELLOW,
            )
            llm_config.context_window = context_window
        if model_wrapper and model_wrapper != llm_config.model_wrapper:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model wrapper {llm_config.model_wrapper} with {model_wrapper}",
                fg=typer.colors.YELLOW,
            )
            llm_config.model_wrapper = model_wrapper
        if model_endpoint and model_endpoint != llm_config.model_endpoint:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint {llm_config.model_endpoint} with {model_endpoint}",
                fg=typer.colors.YELLOW,
            )
            llm_config.model_endpoint = model_endpoint
        if model_endpoint_type and model_endpoint_type != llm_config.model_endpoint_type:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint type {llm_config.model_endpoint_type} with {model_endpoint_type}",
                fg=typer.colors.YELLOW,
            )
            llm_config.model_endpoint_type = model_endpoint_type

        agent_state = AgentState(
            name=agent,
            user_id=user.id,
            persona=persona if persona else config.persona,
            human=human if human else config.human,
            preset=preset if preset else config.preset,
            llm_config=llm_config,
            embedding_config=embedding_config,
        )
        ms.create_agent(agent_state)

        typer.secho(f"->  🤖 Using persona profile '{agent_state.persona}'", fg=typer.colors.WHITE)
        typer.secho(f"->  🧑 Using human profile '{agent_state.human}'", fg=typer.colors.WHITE)

        # Supress llama-index noise
        # TODO(swooders) add persistence manager code? or comment out?
        # with suppress_stdout():
        # TODO: allow configrable state manager (only local is supported right now)
        # persistence_manager = LocalStateManager(agent_config)  # TODO: insert dataset/pre-fill

        # create agent
        try:
            memgpt_agent = presets.create_agent_from_preset(
                agent_state=agent_state,
                interface=interface(),
            )
            save_agent(agent=memgpt_agent, ms=ms)
        except ValueError as e:
            # TODO(swooders) what's the equivalent cleanup code for the new DB refactor?
            typer.secho(f"Failed to create agent from provided information:\n{e}", fg=typer.colors.RED)
            # # Delete the directory of the failed agent
            # try:
            #     # Path to the specific file
            #     agent_config_file = agent_config.agent_config_path

            #     # Check if the file exists
            #     if os.path.isfile(agent_config_file):
            #         # Delete the file
            #         os.remove(agent_config_file)

            #     # Now, delete the directory along with any remaining files in it
            #     agent_save_dir = os.path.join(MEMGPT_DIR, "agents", agent_config.name)
            #     shutil.rmtree(agent_save_dir)
            # except:
            #     typer.secho(f"Failed to delete agent directory during cleanup:\n{e}", fg=typer.colors.RED)
            sys.exit(1)
        typer.secho(f"🎉 Created new agent '{agent_state.name}' (id={agent_state.id})", fg=typer.colors.GREEN)

    # pretty print agent config
    # printd(json.dumps(vars(agent_config), indent=4, sort_keys=True, ensure_ascii=JSON_ENSURE_ASCII))
    # printd(json.dumps(agent_init_state), indent=4, sort_keys=True, ensure_ascii=JSON_ENSURE_ASCII))

    # configure llama index
    original_stdout = sys.stdout  # unfortunate hack required to suppress confusing print statements from llama index
    sys.stdout = io.StringIO()
    embed_model = embedding_model(config=agent_state.embedding_config, user_id=user.id)
    service_context = ServiceContext.from_defaults(
        llm=None, embed_model=embed_model, chunk_size=agent_state.embedding_config.embedding_chunk_size
    )
    set_global_service_context(service_context)
    sys.stdout = original_stdout

    # start event loop
    from memgpt.main import run_agent_loop

    print()  # extra space
    run_agent_loop(memgpt_agent, config, first, ms, no_verify)  # TODO: add back no_verify


def delete_agent(
    agent_name: Annotated[str, typer.Option(help="Specify agent to delete")],
    user_id: Annotated[Optional[str], typer.Option(help="User ID to associate with the agent.")] = None,
):
    """Delete an agent from the database"""
    # use client ID is no user_id provided
    config = MemGPTConfig.load()
    ms = MetadataStore(config)
    if user_id is None:
        user = create_default_user_or_exit(config, ms)
    else:
        user = ms.get_user(user_id=uuid.UUID(user_id))

    try:
        agent = ms.get_agent(agent_name=agent_name, user_id=user.id)
    except Exception as e:
        typer.secho(f"Failed to get agent {agent_name}\n{e}", fg=typer.colors.RED)
        sys.exit(1)

    if agent is None:
        typer.secho(f"Couldn't find agent named '{agent_name}' to delete", fg=typer.colors.RED)
        sys.exit(1)

    confirm = questionary.confirm(f"Are you sure you want to delete agent '{agent_name}' (id={agent.id})?", default=False).ask()
    if confirm is None:
        raise KeyboardInterrupt
    if not confirm:
        typer.secho(f"Cancelled agent deletion '{agent_name}' (id={agent.id})", fg=typer.colors.GREEN)
        return

    try:
        ms.delete_agent(agent_id=agent.id)
        typer.secho(f"🕊️ Successfully deleted agent '{agent_name}' (id={agent.id})", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to delete agent '{agent_name}' (id={agent.id})", fg=typer.colors.RED)
        sys.exit(1)


def attach(
    agent_name: Annotated[str, typer.Option(help="Specify agent to attach data to")],
    data_source: Annotated[str, typer.Option(help="Data source to attach to agent")],
    user_id: uuid.UUID = None,
):
    # use client ID is no user_id provided
    config = MemGPTConfig.load()
    if user_id is None:
        user_id = uuid.UUID(config.anon_clientid)
    try:
        # loads the data contained in data source into the agent's memory
        from memgpt.agent_store.storage import StorageConnector, TableType
        from tqdm import tqdm

        ms = MetadataStore(config)
        agent = ms.get_agent(agent_name=agent_name, user_id=user_id)
        assert agent is not None, f"No agent found under agent_name={agent_name}, user_id={user_id}"
        source = ms.get_source(source_name=data_source, user_id=user_id)
        assert source is not None, f"Source {data_source} does not exist for user {user_id}"

        # get storage connectors
        with suppress_stdout():
            source_storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id=user_id)
            dest_storage = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config, user_id=user_id, agent_id=agent.id)

        size = source_storage.size({"data_source": data_source})
        typer.secho(f"Ingesting {size} passages into {agent.name}", fg=typer.colors.GREEN)
        page_size = 100
        generator = source_storage.get_all_paginated(filters={"data_source": data_source}, page_size=page_size)  # yields List[Passage]
        all_passages = []
        for i in tqdm(range(0, size, page_size)):
            passages = next(generator)

            # need to associated passage with agent (for filtering)
            for passage in passages:
                assert isinstance(passage, Passage), f"Generate yielded bad non-Passage type: {type(passage)}"
                passage.agent_id = agent.id

            # insert into agent archival memory
            dest_storage.insert_many(passages)
            all_passages += passages

        assert size == len(all_passages), f"Expected {size} passages, but only got {len(all_passages)}"

        # save destination storage
        dest_storage.save()

        # attach to agent
        source = ms.get_source(source_name=data_source, user_id=user_id)
        assert source is not None, f"source does not exist for source_name={data_source}, user_id={user_id}"
        source_id = source.id
        ms.attach_source(agent_id=agent.id, source_id=source_id, user_id=user_id)

        total_agent_passages = dest_storage.size()

        typer.secho(
            f"Attached data source {data_source} to agent {agent_name}, consisting of {len(all_passages)}. Agent now has {total_agent_passages} embeddings in archival memory.",
            fg=typer.colors.GREEN,
        )
    except KeyboardInterrupt:
        typer.secho("Operation interrupted by KeyboardInterrupt.", fg=typer.colors.YELLOW)


def version():
    import memgpt

    print(memgpt.__version__)
    return memgpt.__version__
