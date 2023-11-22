import typer
import json
import sys
import io
import logging
import os
from prettytable import PrettyTable
import questionary
import openai

from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

from memgpt.interface import CLIInterface as interface  # for printing to terminal
from memgpt.cli.cli_config import configure
import memgpt.agent as agent
import memgpt.system as system
import memgpt.presets.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
import memgpt.utils as utils
from memgpt.utils import printd
from memgpt.persistence_manager import LocalStateManager
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.agent import Agent
from memgpt.embeddings import embedding_model
from memgpt.openai_tools import (
    configure_azure_support,
    check_azure_embeddings,
)


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

    if not MemGPTConfig.exists():  # if no config, run configure
        if yes:
            # use defaults
            config = MemGPTConfig()
        else:
            # use input
            configure()
            config = MemGPTConfig.load()
    else:  # load config
        config = MemGPTConfig.load()

        # force re-configuration is config is from old version
        if config.memgpt_version is None:  # TODO: eventually add checks for older versions, if config changes again
            typer.secho("MemGPT has been updated to a newer version, so re-running configuration.", fg=typer.colors.YELLOW)
            configure()
            config = MemGPTConfig.load()

    # override with command line arguments
    if debug:
        config.debug = debug
    if no_verify:
        config.no_verify = no_verify

    # determine agent to use, if not provided
    if not yes and not agent:
        agent_files = utils.list_agent_config_files()
        agents = [AgentConfig.load(f).name for f in agent_files]

        if len(agents) > 0 and not any([persona, human, model]):
            select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
            if select_agent:
                agent = questionary.select("Select agent:", choices=agents).ask()

    # configure llama index
    config = MemGPTConfig.load()
    original_stdout = sys.stdout  # unfortunate hack required to suppress confusing print statements from llama index
    sys.stdout = io.StringIO()
    embed_model = embedding_model()
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=config.embedding_chunk_size)
    set_global_service_context(service_context)
    sys.stdout = original_stdout

    # create agent config
    if agent and AgentConfig.exists(agent):  # use existing agent
        typer.secho(f"Using existing agent {agent}", fg=typer.colors.GREEN)
        agent_config = AgentConfig.load(agent)
        printd("State path:", agent_config.save_state_dir())
        printd("Persistent manager path:", agent_config.save_persistence_manager_dir())
        printd("Index path:", agent_config.save_agent_index_dir())
        # persistence_manager = LocalStateManager(agent_config).load() # TODO: implement load
        # TODO: load prior agent state
        if persona and persona != agent_config.persona:
            typer.secho(f"Warning: Overriding existing persona {agent_config.persona} with {persona}", fg=typer.colors.YELLOW)
            agent_config.persona = persona
            # raise ValueError(f"Cannot override {agent_config.name} existing persona {agent_config.persona} with {persona}")
        if human and human != agent_config.human:
            typer.secho(f"Warning: Overriding existing human {agent_config.human} with {human}", fg=typer.colors.YELLOW)
            agent_config.human = human
            # raise ValueError(f"Cannot override {agent_config.name} existing human {agent_config.human} with {human}")

        # Allow overriding model specifics (model, model wrapper, model endpoint IP + type, context_window)
        if model and model != agent_config.model:
            typer.secho(f"Warning: Overriding existing model {agent_config.model} with {model}", fg=typer.colors.YELLOW)
            agent_config.model = model
        if context_window is not None and int(context_window) != agent_config.context_window:
            typer.secho(
                f"Warning: Overriding existing context window {agent_config.context_window} with {context_window}", fg=typer.colors.YELLOW
            )
            agent_config.context_window = context_window
        if model_wrapper and model_wrapper != agent_config.model_wrapper:
            typer.secho(
                f"Warning: Overriding existing model wrapper {agent_config.model_wrapper} with {model_wrapper}", fg=typer.colors.YELLOW
            )
            agent_config.model_wrapper = model_wrapper
        if model_endpoint and model_endpoint != agent_config.model_endpoint:
            typer.secho(
                f"Warning: Overriding existing model endpoint {agent_config.model_endpoint} with {model_endpoint}", fg=typer.colors.YELLOW
            )
            agent_config.model_endpoint = model_endpoint
        if model_endpoint_type and model_endpoint_type != agent_config.model_endpoint_type:
            typer.secho(
                f"Warning: Overriding existing model endpoint type {agent_config.model_endpoint_type} with {model_endpoint_type}",
                fg=typer.colors.YELLOW,
            )
            agent_config.model_endpoint_type = model_endpoint_type

        # Update the agent config with any overrides
        agent_config.save()

        # load existing agent
        memgpt_agent = Agent.load_agent(interface, agent_config)
    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("Creating new agent...", fg=typer.colors.GREEN)
        agent_config = AgentConfig(
            name=agent,
            persona=persona,
            human=human,
            preset=preset,
            model=model,
            model_wrapper=model_wrapper,
            model_endpoint_type=model_endpoint_type,
            model_endpoint=model_endpoint,
            context_window=context_window,
        )

        # TODO: allow configrable state manager (only local is supported right now)
        persistence_manager = LocalStateManager(agent_config)  # TODO: insert dataset/pre-fill

        # save new agent config
        agent_config.save()
        typer.secho(f"Created new agent {agent_config.name}.", fg=typer.colors.GREEN)

        # create agent
        memgpt_agent = presets.use_preset(
            agent_config.preset,
            agent_config,
            agent_config.model,
            utils.get_persona_text(agent_config.persona),
            utils.get_human_text(agent_config.human),
            interface,
            persistence_manager,
        )

    # pretty print agent config
    printd(json.dumps(vars(agent_config), indent=4, sort_keys=True))

    # start event loop
    from memgpt.main import run_agent_loop

    # setup azure if using
    # TODO: cleanup this code
    if config.model_endpoint == "azure":
        configure_azure_support()

    run_agent_loop(memgpt_agent, first, no_verify, config)  # TODO: add back no_verify


def attach(
    agent: str = typer.Option(help="Specify agent to attach data to"),
    data_source: str = typer.Option(help="Data source to attach to avent"),
):
    # loads the data contained in data source into the agent's memory
    from memgpt.connectors.storage import StorageConnector
    from tqdm import tqdm

    agent_config = AgentConfig.load(agent)

    # get storage connectors
    source_storage = StorageConnector.get_storage_connector(name=data_source)
    dest_storage = StorageConnector.get_storage_connector(agent_config=agent_config)

    size = source_storage.size()
    typer.secho(f"Ingesting {size} passages into {agent_config.name}", fg=typer.colors.GREEN)
    page_size = 100
    generator = source_storage.get_all_paginated(page_size=page_size)  # yields List[Passage]
    passages = []
    for i in tqdm(range(0, size, page_size)):
        passages = next(generator)
        dest_storage.insert_many(passages)

    # save destination storage
    dest_storage.save()

    total_agent_passages = dest_storage.size()

    typer.secho(
        f"Attached data source {data_source} to agent {agent}, consisting of {len(passages)}. Agent now has {total_agent_passages} embeddings in archival memory.",
        fg=typer.colors.GREEN,
    )


def version():
    import memgpt

    print(memgpt.__version__)
    return memgpt.__version__
