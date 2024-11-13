import logging
import sys
from enum import Enum
from typing import Annotated, Optional

import questionary
import typer

import letta.utils as utils
from letta import create_client
from letta.agent import Agent, save_agent
from letta.config import LettaConfig
from letta.constants import CLI_WARNING_PREFIX, LETTA_DIR, MIN_CONTEXT_WINDOW
from letta.local_llm.constants import ASSISTANT_MESSAGE_CLI_SYMBOL
from letta.log import get_logger
from letta.metadata import MetadataStore
from letta.schemas.enums import OptionState
from letta.schemas.memory import ChatMemory, Memory
from letta.server.server import logger as server_logger

# from letta.interface import CLIInterface as interface  # for printing to terminal
from letta.streaming_interface import (
    StreamingRefreshCLIInterface as interface,  # for printing to terminal
)
from letta.utils import open_folder_in_explorer, printd

logger = get_logger(__name__)


def open_folder():
    """Open a folder viewer of the Letta home directory"""
    try:
        print(f"Opening home folder: {LETTA_DIR}")
        open_folder_in_explorer(LETTA_DIR)
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
    ade: Annotated[bool, typer.Option(help="Allows remote access")] = False,
    secure: Annotated[bool, typer.Option(help="Adds simple security access")] = False,
):
    """Launch a Letta server process"""
    if type == ServerChoice.rest_api:
        pass

        # if LettaConfig.exists():
        #    config = LettaConfig.load()
        #    MetadataStore(config)
        #    _ = create_client()  # triggers user creation
        # else:
        #    typer.secho(f"No configuration exists. Run letta configure before starting the server.", fg=typer.colors.RED)
        #    sys.exit(1)

        try:
            from letta.server.rest_api.app import start_server

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
    """Start chatting with an Letta agent

    Example usage: `letta run --agent myagent --data-source mydata --persona mypersona --human myhuman --model gpt-3.5-turbo`

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

    # load config file
    config = LettaConfig.load()

    # read user id from config
    ms = MetadataStore(config)
    client = create_client()
    server = client.server

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
        printd("Agent state:", agent_state.name)
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
        tools = [server.tool_manager.get_tool_by_name(tool_name=tool_name, actor=client.user) for tool_name in agent_state.tools]
        letta_agent = Agent(agent_state=agent_state, interface=interface(), tools=tools)

    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)

        agent_name = agent if agent else utils.create_random_username()

        # create agent
        client = create_client()

        # choose from list of llm_configs
        llm_configs = client.list_llm_configs()
        llm_options = [llm_config.model for llm_config in llm_configs]
        llm_choices = [questionary.Choice(title=llm_config.pretty_print(), value=llm_config) for llm_config in llm_configs]

        # select model
        if len(llm_options) == 0:
            raise ValueError("No LLM models found. Please enable a provider.")
        elif len(llm_options) == 1:
            llm_model_name = llm_options[0]
        else:
            llm_model_name = questionary.select("Select LLM model:", choices=llm_choices).ask().model
        llm_config = [llm_config for llm_config in llm_configs if llm_config.model == llm_model_name][0]

        # option to override context window
        if llm_config.context_window is not None:
            context_window_validator = lambda x: x.isdigit() and int(x) > MIN_CONTEXT_WINDOW and int(x) <= llm_config.context_window
            context_window_input = questionary.text(
                "Select LLM context window limit (hit enter for default):",
                default=str(llm_config.context_window),
                validate=context_window_validator,
            ).ask()
            if context_window_input is not None:
                llm_config.context_window = int(context_window_input)
            else:
                sys.exit(1)

        # choose form list of embedding configs
        embedding_configs = client.list_embedding_configs()
        embedding_options = [embedding_config.embedding_model for embedding_config in embedding_configs]

        embedding_choices = [
            questionary.Choice(title=embedding_config.pretty_print(), value=embedding_config) for embedding_config in embedding_configs
        ]

        # select model
        if len(embedding_options) == 0:
            raise ValueError("No embedding models found. Please enable a provider.")
        elif len(embedding_options) == 1:
            embedding_model_name = embedding_options[0]
        else:
            embedding_model_name = questionary.select("Select embedding model:", choices=embedding_choices).ask().embedding_model
        embedding_config = [
            embedding_config for embedding_config in embedding_configs if embedding_config.embedding_model == embedding_model_name
        ][0]

        human_obj = client.get_human(client.get_human_id(name=human))
        persona_obj = client.get_persona(client.get_persona_id(name=persona))
        if human_obj is None:
            typer.secho(f"Couldn't find human {human} in database, please run `letta add human`", fg=typer.colors.RED)
            sys.exit(1)
        if persona_obj is None:
            typer.secho(f"Couldn't find persona {persona} in database, please run `letta add persona`", fg=typer.colors.RED)
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
        metadata = {"human": human_obj.template_name, "persona": persona_obj.template_name}

        typer.secho(f"->  {ASSISTANT_MESSAGE_CLI_SYMBOL} Using persona profile: '{persona_obj.template_name}'", fg=typer.colors.WHITE)
        typer.secho(f"->  🧑 Using human profile: '{human_obj.template_name}'", fg=typer.colors.WHITE)

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
        tools = [server.tool_manager.get_tool_by_name(tool_name, actor=client.user) for tool_name in agent_state.tools]

        letta_agent = Agent(
            interface=interface(),
            agent_state=agent_state,
            tools=tools,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
        )
        save_agent(agent=letta_agent, ms=ms)
        typer.secho(f"🎉 Created new agent '{letta_agent.agent_state.name}' (id={letta_agent.agent_state.id})", fg=typer.colors.GREEN)

    # start event loop
    from letta.main import run_agent_loop

    print()  # extra space
    run_agent_loop(
        letta_agent=letta_agent,
        config=config,
        first=first,
        ms=ms,
        no_verify=no_verify,
        stream=stream,
    )  # TODO: add back no_verify


def delete_agent(
    agent_name: Annotated[str, typer.Option(help="Specify agent to delete")],
):
    """Delete an agent from the database"""
    # use client ID is no user_id provided
    config = LettaConfig.load()
    MetadataStore(config)
    client = create_client()
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
    import letta

    print(letta.__version__)
    return letta.__version__
