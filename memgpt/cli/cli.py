import typer
import uuid
import json
import requests
import sys
import shutil
import io
import logging
import questionary
from pathlib import Path
import os
import subprocess
from enum import Enum

from llama_index import set_global_service_context
from llama_index import ServiceContext

from memgpt.interface import CLIInterface as interface  # for printing to terminal
from memgpt.cli.cli_config import configure
import memgpt.presets.presets as presets
import memgpt.utils as utils
from memgpt.utils import printd, open_folder_in_explorer, suppress_stdout
from memgpt.persistence_manager import LocalStateManager
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR, CLI_WARNING_PREFIX
from memgpt.agent import Agent
from memgpt.embeddings import embedding_model
from memgpt.server.constants import WS_DEFAULT_PORT, REST_DEFAULT_PORT
from memgpt.data_types import AgentState, LLMConfig, EmbeddingConfig, User
from memgpt.metadata import MetadataStore


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

    if modified:
        printd(f"Saving new config file.")
        old_config.save()
        typer.secho(f"📖 MemGPT configuration file updated!", fg=typer.colors.GREEN)
        typer.secho(f"🧠 model\t-> {old_config.model}\n🖥️  endpoint\t-> {old_config.model_endpoint}", fg=typer.colors.GREEN)
        return True
    else:
        typer.secho(f"📖 MemGPT configuration file unchanged.", fg=typer.colors.WHITE)
        typer.secho(f"🧠 model\t-> {old_config.model}\n🖥️  endpoint\t-> {old_config.model_endpoint}", fg=typer.colors.WHITE)
        return False


def quickstart(
    backend: QuickstartChoice = typer.Option("memgpt", help="Quickstart setup backend"),
    latest: bool = typer.Option(False, "--latest", help="Use --latest to pull the latest config from online"),
    debug: bool = typer.Option(False, "--debug", help="Use --debug to enable debugging output"),
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
                    with open(backup_config_path, "r") as file:
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
                with open(backup_config_path, "r") as file:
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
            api_key = questionary.text("Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):").ask()

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
                # Add the API key
                config["openai_key"] = api_key
                config_was_modified = set_config_with_dict(config)
            else:
                typer.secho(f"Failed to download config from {url}. Status code: {response.status_code}", fg=typer.colors.RED)

                # Load the file from the relative path
                script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
                backup_config_path = os.path.join(script_dir, "..", "configs", "openai.json")
                try:
                    with open(backup_config_path, "r") as file:
                        backup_config = json.load(file)
                        backup_config["openai_key"] = api_key
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
                with open(backup_config_path, "r") as file:
                    backup_config = json.load(file)
                    backup_config["openai_key"] = api_key
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


def server(
    type: ServerChoice = typer.Option("rest", help="Server to run"),
    port: int = typer.Option(None, help="Port to run the server on"),
    host: str = typer.Option(None, help="Host to run the server on (default to localhost)"),
):
    """Launch a MemGPT server process"""

    if type == ServerChoice.rest_api:
        if port is None:
            port = REST_DEFAULT_PORT

        # Change to the desired directory
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent

        server_directory = os.path.join(script_dir.parent, "server", "rest_api")
        if host is None:
            command = f"uvicorn server:app --reload --port {port}"
        else:
            command = f"uvicorn server:app --reload --port {port} --host {host}"

        # Run the command
        print(f"Running REST server: {command} (inside {server_directory})")

        try:
            # Start the subprocess in a new session
            process = subprocess.Popen(command, shell=True, start_new_session=True, cwd=server_directory)
            process.wait()
        except KeyboardInterrupt:
            # Handle CTRL-C
            print("Terminating the server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                print("Server terminated with kill()")
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
        print(f"Running WS (websockets) server: {command} (inside {server_directory})")

        try:
            # Start the subprocess in a new session
            process = subprocess.Popen(command, shell=True, start_new_session=True, cwd=server_directory)
            process.wait()
        except KeyboardInterrupt:
            # Handle CTRL-C
            print("Terminating the server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                print("Server terminated with kill()")
            sys.exit(0)


def run(
    persona: str = typer.Option(None, help="Specify persona"),
    agent: str = typer.Option(None, help="Specify agent save file"),
    human: str = typer.Option(None, help="Specify human"),
    preset: str = typer.Option(None, help="Specify preset"),
    # model flags
    model: str = typer.Option(None, help="Specify the LLM model"),
    model_wrapper: str = typer.Option(None, help="Specify the LLM model wrapper"),
    model_endpoint: str = typer.Option(None, help="Specify the LLM model endpoint"),
    model_endpoint_type: str = typer.Option(None, help="Specify the LLM model endpoint type"),
    context_window: int = typer.Option(None, help="The context window of the LLM you are using (e.g. 8k for most Mistral 7B variants)"),
    # other
    first: bool = typer.Option(False, "--first", help="Use --first to send the first message in the sequence"),
    strip_ui: bool = typer.Option(False, help="Remove all the bells and whistles in CLI output (helpful for testing)"),
    debug: bool = typer.Option(False, "--debug", help="Use --debug to enable debugging output"),
    no_verify: bool = typer.Option(False, help="Bypass message verification"),
    yes: bool = typer.Option(False, "-y", help="Skip confirmation prompt and use defaults"),
):
    """Start chatting with an MemGPT agent

    Example usage: `memgpt run --agent myagent --data-source mydata --persona mypersona --human myhuman --model gpt-3.5-turbo`

    :param persona: Specify persona
    :param agent: Specify agent name (will load existing state if the agent exists, or create a new one with that name)
    :param human: Specify human
    :param model: Specify the LLM model

    """

    # setup logger
    utils.DEBUG = debug
    logging.getLogger().setLevel(logging.CRITICAL)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
    user_id = uuid.UUID(config.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        ms.create_user(User(id=user_id))
        user = ms.get_user(user_id=user_id)
        if user is None:
            typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
            sys.exit(1)

    # override with command line arguments
    if debug:
        config.debug = debug
    if no_verify:
        config.no_verify = no_verify

    # determine agent to use, if not provided
    if not yes and not agent:
        agents = ms.list_agents(user_id=user.id)
        agents = [a.name for a in agents]

        if len(agents) > 0 and not any([persona, human, model]):
            print()
            select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
            if select_agent:
                agent = questionary.select("Select agent:", choices=agents).ask()

    # create agent config
    if agent and ms.get_agent(agent_name=agent, user_id=user.id):  # use existing agent
        typer.secho(f"\n🔁 Using existing agent {agent}", fg=typer.colors.GREEN)
        # agent_config = AgentConfig.load(agent)
        agent_state = ms.get_agent(agent_name=agent, user_id=user_id)
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
        memgpt_agent = Agent(agent_state, interface=interface)

    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)

        if agent is None:
            # determine agent name
            # agent_count = len(ms.list_agents(user_id=user.id))
            # agent = f"agent_{agent_count}"
            agent = utils.create_random_username()

        agent_state = AgentState(
            name=agent,
            user_id=user.id,
            persona=persona if persona else user.default_persona,
            human=human if human else user.default_human,
            preset=preset if preset else user.default_preset,
            llm_config=user.default_llm_config,
            embedding_config=user.default_embedding_config,
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
                interface=interface,
            )
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
        typer.secho(f"🎉 Created new agent '{agent_state.name}'", fg=typer.colors.GREEN)

    # pretty print agent config
    # printd(json.dumps(vars(agent_config), indent=4, sort_keys=True, ensure_ascii=False))
    # printd(json.dumps(agent_init_state), indent=4, sort_keys=True, ensure_ascii=False))

    # configure llama index
    original_stdout = sys.stdout  # unfortunate hack required to suppress confusing print statements from llama index
    sys.stdout = io.StringIO()
    embed_model = embedding_model(config=agent_state.embedding_config, user_id=user.id)
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=config.embedding_chunk_size)
    set_global_service_context(service_context)
    sys.stdout = original_stdout

    # start event loop
    from memgpt.main import run_agent_loop

    print()  # extra space
    run_agent_loop(memgpt_agent, config, first, no_verify)  # TODO: add back no_verify


def attach(
    agent: str = typer.Option(help="Specify agent to attach data to"),
    data_source: str = typer.Option(help="Data source to attach to avent"),
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
        agent = ms.get_agent(agent_name=agent, user_id=user_id)
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
        passages = []
        for i in tqdm(range(0, size, page_size)):
            passages = next(generator)
            print("inserting", passages)

            # need to associated passage with agent (for filtering)
            for passage in passages:
                passage.agent_id = agent.id

            # insert into agent archival memory
            dest_storage.insert_many(passages)

        # save destination storage
        dest_storage.save()

        # attach to agent
        source_id = ms.get_source(source_name=data_source, user_id=user_id).id
        ms.attach_source(agent_id=agent.id, source_id=source_id, user_id=user_id)

        total_agent_passages = dest_storage.size()

        typer.secho(
            f"Attached data source {data_source} to agent {agent}, consisting of {len(passages)}. Agent now has {total_agent_passages} embeddings in archival memory.",
            fg=typer.colors.GREEN,
        )
    except KeyboardInterrupt:
        typer.secho("Operation interrupted by KeyboardInterrupt.", fg=typer.colors.YELLOW)


def version():
    import memgpt

    print(memgpt.__version__)
    return memgpt.__version__
