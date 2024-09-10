import json
import logging
import os
import sys
from enum import Enum
from typing import Annotated, Optional

import questionary
import requests
import typer

import memgpt.utils as utils
from memgpt import create_client
from memgpt.agent import Agent, save_agent
from memgpt.cli.cli_config import configure
from memgpt.config import MemGPTConfig
from memgpt.constants import CLI_WARNING_PREFIX, MEMGPT_DIR
from memgpt.credentials import MemGPTCredentials
from memgpt.log import get_logger
from memgpt.metadata import MetadataStore
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.enums import OptionState
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.memory import ChatMemory, Memory
from memgpt.server.server import logger as server_logger

# from memgpt.interface import CLIInterface as interface  # for printing to terminal
from memgpt.streaming_interface import (
    StreamingRefreshCLIInterface as interface,  # for printing to terminal
)
from memgpt.utils import open_folder_in_explorer, printd

logger = get_logger(__name__)


class QuickstartChoice(Enum):
    openai = "openai"
    # azure = "azure"
    memgpt_hosted = "memgpt"
    anthropic = "anthropic"


def str_to_quickstart_choice(choice_str: str) -> QuickstartChoice:
    try:
        return QuickstartChoice[choice_str]
    except KeyError:
        valid_options = [choice.name for choice in QuickstartChoice]
        raise ValueError(f"{choice_str} is not a valid QuickstartChoice. Valid options are: {valid_options}")


def set_config_with_dict(new_config: dict) -> (MemGPTConfig, bool):
    """_summary_

    Args:
        new_config (dict): Dict of new config values

    Returns:
        new_config MemGPTConfig, modified (bool): Returns the new config and a boolean indicating if the config was modified
    """
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
    if old_config.default_embedding_config:
        for k, v in vars(old_config.default_embedding_config).items():
            if k in new_config:
                if v != new_config[k]:
                    printd(f"Replacing config {k}: {v} -> {new_config[k]}")
                    modified = True
                    # old_config[k] = new_config[k]
                    setattr(old_config.default_embedding_config, k, new_config[k])
                else:
                    printd(f"Skipping new config {k}: {v} == {new_config[k]}")
    else:
        modified = True
        fields = ["embedding_model", "embedding_dim", "embedding_chunk_size", "embedding_endpoint", "embedding_endpoint_type"]
        args = {}
        for field in fields:
            if field in new_config:
                args[field] = new_config[field]
                printd(f"Setting new config {field}: {new_config[field]}")
        old_config.default_embedding_config = EmbeddingConfig(**args)

    # update llm config
    if old_config.default_llm_config:
        for k, v in vars(old_config.default_llm_config).items():
            if k in new_config:
                if v != new_config[k]:
                    printd(f"Replacing config {k}: {v} -> {new_config[k]}")
                    modified = True
                    # old_config[k] = new_config[k]
                    setattr(old_config.default_llm_config, k, new_config[k])
                else:
                    printd(f"Skipping new config {k}: {v} == {new_config[k]}")
    else:
        modified = True
        fields = ["model", "model_endpoint", "model_endpoint_type", "model_wrapper", "context_window"]
        args = {}
        for field in fields:
            if field in new_config:
                args[field] = new_config[field]
                printd(f"Setting new config {field}: {new_config[field]}")
        old_config.default_llm_config = LLMConfig(**args)
    return (old_config, modified)


def quickstart(
    backend: Annotated[QuickstartChoice, typer.Option(help="Quickstart setup backend")] = "memgpt",
    latest: Annotated[bool, typer.Option(help="Use --latest to pull the latest config from online")] = False,
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
    terminal: bool = True,
):
    """Set the base config file with a single command

    This function and `configure` should be the ONLY places where MemGPTConfig.save() is called.
    """

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
            url = "https://raw.githubusercontent.com/cpacker/MemGPT/main/configs/memgpt_hosted.json"
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response content as JSON
                config = response.json()
                # Output a success message and the first few items in the dictionary as a sample
                printd("JSON config file downloaded successfully.")
                new_config, config_was_modified = set_config_with_dict(config)
            else:
                typer.secho(f"Failed to download config from {url}. Status code: {response.status_code}", fg=typer.colors.RED)

                # Load the file from the relative path
                script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
                backup_config_path = os.path.join(script_dir, "..", "configs", "memgpt_hosted.json")
                try:
                    with open(backup_config_path, "r", encoding="utf-8") as file:
                        backup_config = json.load(file)
                    printd("Loaded backup config file successfully.")
                    new_config, config_was_modified = set_config_with_dict(backup_config)
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
                new_config, config_was_modified = set_config_with_dict(backup_config)
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
            url = "https://raw.githubusercontent.com/cpacker/MemGPT/main/configs/openai.json"
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response content as JSON
                config = response.json()
                # Output a success message and the first few items in the dictionary as a sample
                new_config, config_was_modified = set_config_with_dict(config)
            else:
                typer.secho(f"Failed to download config from {url}. Status code: {response.status_code}", fg=typer.colors.RED)

                # Load the file from the relative path
                script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
                backup_config_path = os.path.join(script_dir, "..", "configs", "openai.json")
                try:
                    with open(backup_config_path, "r", encoding="utf-8") as file:
                        backup_config = json.load(file)
                    printd("Loaded backup config file successfully.")
                    new_config, config_was_modified = set_config_with_dict(backup_config)
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
                new_config, config_was_modified = set_config_with_dict(backup_config)
            except FileNotFoundError:
                typer.secho(f"Config file not found at {backup_config_path}", fg=typer.colors.RED)
                return

    elif backend == QuickstartChoice.anthropic:
        # Make sure we have an API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        while api_key is None or len(api_key) == 0:
            # Ask for API key as input
            api_key = questionary.password("Enter your Anthropic API key:").ask()
        credentials.anthropic_key = api_key
        credentials.save()

        script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
        backup_config_path = os.path.join(script_dir, "..", "configs", "anthropic.json")
        try:
            with open(backup_config_path, "r", encoding="utf-8") as file:
                backup_config = json.load(file)
            printd("Loaded config file successfully.")
            new_config, config_was_modified = set_config_with_dict(backup_config)
        except FileNotFoundError:
            typer.secho(f"Config file not found at {backup_config_path}", fg=typer.colors.RED)
            return

    else:
        raise NotImplementedError(backend)

    if config_was_modified:
        printd(f"Saving new config file.")
        new_config.save()
        typer.secho(f"📖 MemGPT configuration file updated!", fg=typer.colors.GREEN)
        typer.secho(
            "\n".join(
                [
                    f"🧠 model\t-> {new_config.default_llm_config.model}",
                    f"🖥️  endpoint\t-> {new_config.default_llm_config.model_endpoint}",
                ]
            ),
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(f"📖 MemGPT configuration file unchanged.", fg=typer.colors.WHITE)
        typer.secho(
            "\n".join(
                [
                    f"🧠 model\t-> {new_config.default_llm_config.model}",
                    f"🖥️  endpoint\t-> {new_config.default_llm_config.model_endpoint}",
                ]
            ),
            fg=typer.colors.WHITE,
        )

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


def server(
    type: Annotated[ServerChoice, typer.Option(help="Server to run")] = "rest",
    port: Annotated[Optional[int], typer.Option(help="Port to run the server on")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host to run the server on (default to localhost)")] = None,
    debug: Annotated[bool, typer.Option(help="Turn debugging output on")] = False,
):
    """Launch a MemGPT server process"""

    if type == ServerChoice.rest_api:
        pass

        if MemGPTConfig.exists():
            config = MemGPTConfig.load()
            MetadataStore(config)
            _ = create_client()  # triggers user creation
        else:
            typer.secho(f"No configuration exists. Run memgpt configure before starting the server.", fg=typer.colors.RED)
            sys.exit(1)

        try:
            from memgpt.server.rest_api.app import start_server

            start_server(port=port, host=host, debug=debug)

        except KeyboardInterrupt:
            # Handle CTRL-C
            typer.secho("Terminating the server...")
            sys.exit(0)

    elif type == ServerChoice.ws_api:
        raise NotImplementedError("WS suppport deprecated")


def run(
    persona: Annotated[Optional[str], typer.Option(help="Specify persona")] = None,
    agent: Annotated[Optional[str], typer.Option(help="Specify agent name")] = None,
    human: Annotated[Optional[str], typer.Option(help="Specify human")] = None,
    system: Annotated[Optional[str], typer.Option(help="Specify system prompt (raw text)")] = None,
    system_file: Annotated[Optional[str], typer.Option(help="Specify raw text file containing system prompt")] = None,
    # model flags
    model: Annotated[Optional[str], typer.Option(help="Specify the LLM model")] = None,
    model_wrapper: Annotated[Optional[str], typer.Option(help="Specify the LLM model wrapper")] = None,
    model_endpoint: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint")] = None,
    model_endpoint_type: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint type")] = None,
    context_window: Annotated[
        Optional[int], typer.Option(help="The context window of the LLM you are using (e.g. 8k for most Mistral 7B variants)")
    ] = None,
    core_memory_limit: Annotated[
        Optional[int], typer.Option(help="The character limit to each core-memory section (human/persona).")
    ] = 2000,
    # other
    first: Annotated[bool, typer.Option(help="Use --first to send the first message in the sequence")] = False,
    strip_ui: Annotated[bool, typer.Option(help="Remove all the bells and whistles in CLI output (helpful for testing)")] = False,
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
    no_verify: Annotated[bool, typer.Option(help="Bypass message verification")] = False,
    yes: Annotated[bool, typer.Option("-y", help="Skip confirmation prompt and use defaults")] = False,
    # streaming
    stream: Annotated[bool, typer.Option(help="Enables message streaming in the CLI (if the backend supports it)")] = False,
    # whether or not to put the inner thoughts inside the function args
    no_content: Annotated[
        OptionState, typer.Option(help="Set to 'yes' for LLM APIs that omit the `content` field during tool calling")
    ] = OptionState.DEFAULT,
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
        server_logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)
        server_logger.setLevel(logging.CRITICAL)

    # from memgpt.migrate import (
    #    VERSION_CUTOFF,
    #    config_is_compatible,
    #    wipe_config_and_reconfigure,
    # )

    # if not config_is_compatible(allow_empty=True):
    #    typer.secho(f"\nYour current config file is incompatible with MemGPT versions later than {VERSION_CUTOFF}\n", fg=typer.colors.RED)
    #    choices = [
    #        "Run the full config setup (recommended)",
    #        "Create a new config using defaults",
    #        "Cancel",
    #    ]
    #    selection = questionary.select(
    #        f"To use MemGPT, you must either downgrade your MemGPT version (<= {VERSION_CUTOFF}), or regenerate your config. Would you like to proceed?",
    #        choices=choices,
    #        default=choices[0],
    #    ).ask()
    #    if selection == choices[0]:
    #        try:
    #            wipe_config_and_reconfigure()
    #        except Exception as e:
    #            typer.secho(f"Fresh config generation failed - error:\n{e}", fg=typer.colors.RED)
    #            raise
    #    elif selection == choices[1]:
    #        try:
    #            # Don't create a config, so that the next block of code asking about quickstart is run
    #            wipe_config_and_reconfigure(run_configure=False, create_config=False)
    #        except Exception as e:
    #            typer.secho(f"Fresh config generation failed - error:\n{e}", fg=typer.colors.RED)
    #            raise
    #    else:
    #        typer.secho("MemGPT config regeneration cancelled", fg=typer.colors.RED)
    #        raise KeyboardInterrupt()

    #    typer.secho("Note: if you would like to migrate old agents to the new release, please run `memgpt migrate`!", fg=typer.colors.GREEN)

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

    # read user id from config
    ms = MetadataStore(config)
    client = create_client()
    client.user_id

    # determine agent to use, if not provided
    if not yes and not agent:
        agents = client.list_agents()
        agents = [a.name for a in agents]

        if len(agents) > 0:
            print()
            select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
            if select_agent is None:
                raise KeyboardInterrupt
            if select_agent:
                agent = questionary.select("Select agent:", choices=agents).ask()

    # create agent config
    if agent:
        agent_id = client.get_agent_id(agent)
        agent_state = client.get_agent(agent_id)
    else:
        agent_state = None
    human = human if human else config.human
    persona = persona if persona else config.persona
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

        # NOTE: commented out because this seems dangerous - instead users should use /systemswap when in the CLI
        # # user specified a new system prompt
        # if system:
        #     # NOTE: agent_state.system is the ORIGINAL system prompt,
        #     #       whereas agent_state.state["system"] is the LATEST system prompt
        #     existing_system_prompt = agent_state.state["system"] if "system" in agent_state.state else None
        #     if existing_system_prompt != system:
        #         # override
        #         agent_state.state["system"] = system

        # Update the agent with any overrides
        agent_state = client.update_agent(
            agent_id=agent_state.id,
            name=agent_state.name,
            llm_config=agent_state.llm_config,
            embedding_config=agent_state.embedding_config,
        )

        # create agent
        memgpt_agent = Agent(agent_state=agent_state, interface=interface(), tools=tools)

    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)

        agent_name = agent if agent else utils.create_random_username()
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

        # create agent
        client = create_client()
        human_obj = client.get_human(client.get_human_id(name=human))
        persona_obj = client.get_persona(client.get_persona_id(name=persona))
        if human_obj is None:
            typer.secho(f"Couldn't find human {human} in database, please run `memgpt add human`", fg=typer.colors.RED)
            sys.exit(1)
        if persona_obj is None:
            typer.secho(f"Couldn't find persona {persona} in database, please run `memgpt add persona`", fg=typer.colors.RED)
            sys.exit(1)

        if system_file:
            try:
                with open(system_file, "r", encoding="utf-8") as file:
                    system = file.read().strip()
                    printd("Loaded system file successfully.")
            except FileNotFoundError:
                typer.secho(f"System file not found at {system_file}", fg=typer.colors.RED)
        system_prompt = system if system else None

        memory = ChatMemory(human=human_obj.value, persona=persona_obj.value, limit=core_memory_limit)
        metadata = {"human": human_obj.name, "persona": persona_obj.name}

        typer.secho(f"->  🤖 Using persona profile: '{persona_obj.name}'", fg=typer.colors.WHITE)
        typer.secho(f"->  🧑 Using human profile: '{human_obj.name}'", fg=typer.colors.WHITE)

        # add tools
        agent_state = client.create_agent(
            name=agent_name,
            system=system_prompt,
            embedding_config=embedding_config,
            llm_config=llm_config,
            memory=memory,
            metadata=metadata,
        )
        assert isinstance(agent_state.memory, Memory), f"Expected Memory, got {type(agent_state.memory)}"
        typer.secho(f"->  🛠️  {len(agent_state.tools)} tools: {', '.join([t for t in agent_state.tools])}", fg=typer.colors.WHITE)
        tools = [ms.get_tool(tool_name, user_id=client.user_id) for tool_name in agent_state.tools]

        memgpt_agent = Agent(
            interface=interface(),
            agent_state=agent_state,
            tools=tools,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
        )
        save_agent(agent=memgpt_agent, ms=ms)
        typer.secho(f"🎉 Created new agent '{memgpt_agent.agent_state.name}' (id={memgpt_agent.agent_state.id})", fg=typer.colors.GREEN)

    # start event loop
    from memgpt.main import run_agent_loop

    print()  # extra space
    run_agent_loop(
        memgpt_agent=memgpt_agent,
        config=config,
        first=first,
        ms=ms,
        no_verify=no_verify,
        stream=stream,
        inner_thoughts_in_kwargs=no_content,
    )  # TODO: add back no_verify


def delete_agent(
    agent_name: Annotated[str, typer.Option(help="Specify agent to delete")],
    user_id: Annotated[Optional[str], typer.Option(help="User ID to associate with the agent.")] = None,
):
    """Delete an agent from the database"""
    # use client ID is no user_id provided
    config = MemGPTConfig.load()
    MetadataStore(config)
    client = create_client(user_id=user_id)
    agent = client.get_agent_by_name(agent_name)
    if not agent:
        typer.secho(f"Couldn't find agent named '{agent_name}' to delete", fg=typer.colors.RED)
        sys.exit(1)

    confirm = questionary.confirm(f"Are you sure you want to delete agent '{agent_name}' (id={agent.id})?", default=False).ask()
    if confirm is None:
        raise KeyboardInterrupt
    if not confirm:
        typer.secho(f"Cancelled agent deletion '{agent_name}' (id={agent.id})", fg=typer.colors.GREEN)
        return

    try:
        # delete the agent
        client.delete_agent(agent.id)
        typer.secho(f"🕊️ Successfully deleted agent '{agent_name}' (id={agent.id})", fg=typer.colors.GREEN)
    except Exception:
        typer.secho(f"Failed to delete agent '{agent_name}' (id={agent.id})", fg=typer.colors.RED)
        sys.exit(1)


def version():
    import memgpt

    print(memgpt.__version__)
    return memgpt.__version__
