import typer
import sys
import io
import logging
import asyncio
import os
from prettytable import PrettyTable
import questionary
import openai

from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

import memgpt.interface  # for printing to terminal
from memgpt.cli.cli_config import configure
import memgpt.agent as agent
import memgpt.system as system
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
import memgpt.utils as utils
from memgpt.utils import printd
from memgpt.persistence_manager import LocalStateManager
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.agent import AgentAsync
from memgpt.embeddings import embedding_model
from memgpt.openai_tools import (
    configure_azure_support,
    check_azure_embeddings,
)


def run(
    persona: str = typer.Option(None, help="Specify persona"),
    agent: str = typer.Option(None, help="Specify agent save file"),
    human: str = typer.Option(None, help="Specify human"),
    model: str = typer.Option(None, help="Specify the LLM model"),
    preset: str = typer.Option(None, help="Specify preset"),
    data_source: str = typer.Option(None, help="Specify data source to attach to agent"),
    first: bool = typer.Option(False, "--first", help="Use --first to send the first message in the sequence"),
    strip_ui: bool = typer.Option(False, "--strip_ui", help="Remove all the bells and whistles in CLI output (helpful for testing)"),
    debug: bool = typer.Option(False, "--debug", help="Use --debug to enable debugging output"),
    no_verify: bool = typer.Option(False, "--no_verify", help="Bypass message verification"),
    yes: bool = typer.Option(False, "-y", help="Skip confirmation prompt and use defaults"),
):
    """Start chatting with an MemGPT agent

    Example usage: `memgpt run --agent myagent --data-source mydata --persona mypersona --human myhuman --model gpt-3.5-turbo`

    :param persona: Specify persona
    :param agent: Specify agent name (will load existing state if the agent exists, or create a new one with that name)
    :param human: Specify human
    :param model: Specify the LLM model
    :param data_source: Specify data source to attach to agent (if new agent is being created)

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

    # override with command line arguments
    if debug:
        config.debug = debug
    if no_verify:
        config.no_verify = no_verify

    # determine agent to use, if not provided
    if not yes and not agent:
        agent_files = utils.list_agent_config_files()
        agents = [AgentConfig.load(f).name for f in agent_files]

        if len(agents) > 0:
            select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
            if select_agent:
                agent = questionary.select("Select agent:", choices=agents).ask()

    # configure llama index
    config = MemGPTConfig.load()
    original_stdout = sys.stdout  # unfortunate hack required to suppress confusing print statements from llama index
    sys.stdout = io.StringIO()
    embed_model = embedding_model(config)
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
        assert not any(
            [persona, human, model]
        ), f"Cannot override existing agent state with command line arguments: {persona}, {human}, {model}"

        # load existing agent
        memgpt_agent = AgentAsync.load_agent(memgpt.interface, agent_config)
    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("Creating new agent...", fg=typer.colors.GREEN)
        agent_config = AgentConfig(
            name=agent if agent else None,
            persona=persona if persona else config.default_persona,
            human=human if human else config.default_human,
            model=model if model else config.model,
            preset=preset if preset else config.preset,
        )

        # attach data source to agent
        agent_config.attach_data_source(data_source)

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
            memgpt.interface,
            persistence_manager,
        )

    # start event loop
    from memgpt.main import run_agent_loop

    # setup azure if using
    # TODO: cleanup this code
    if config.model_endpoint == "azure":
        configure_azure_support()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agent_loop(memgpt_agent, first, no_verify, config, strip_ui))  # TODO: add back no_verify
