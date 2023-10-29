import typer
import logging
import asyncio
import os
from rich.console import Console
from prettytable import PrettyTable
import questionary
import openai

console = Console()

# import memgpt
# from memgpt.cli.cli_load import app as connector_app
# from memgpt.cli.cli_config import app as configure_app
# import memgpt.cli.cli_load
# import memgpt.cli.cli_config
import memgpt.interface  # for printing to terminal
import memgpt.agent as agent
import memgpt.system as system
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
import memgpt.utils as utils
from memgpt.persistence_manager import LocalStateManager
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.agent import AgentAsync


# app = typer.Typer()
# metadata_app = typer.Typer()
# app.add_typer(metadata_app, name="list")
# app.add_typer(memgpt.cli.cli_load.app, name="load")
# app.add_typer(memgpt.cli.cli_config.app)


# @app.command("run")
def run(
    persona: str = typer.Option(None, help="Specify persona"),
    agent: str = typer.Option(None, help="Specify agent save file"),
    human: str = typer.Option(None, help="Specify human"),
    model: str = typer.Option(None, help="Specify the LLM model"),
    data_source: str = typer.Option(None, help="Specify data source to attach to agent"),
    first: bool = typer.Option(False, "--first", help="Use --first to send the first message in the sequence"),
    debug: bool = typer.Option(False, "--debug", help="Use --debug to enable debugging output"),
    no_verify: bool = typer.Option(False, "--no_verify", help="Bypass message verification"),
):
    print("running command")

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

    # if no config, run configure
    if not MemGPTConfig.exists():
        print("No config found, running configure...")
        memgpt.cli_configure.configure()

    # load config defaults
    config = MemGPTConfig.load()

    # override with command line arguments
    if debug:
        config.debug = debug
    if no_verify:
        config.no_verify = no_verify

    # create agent config
    if agent and AgentConfig.exists(agent):  # use existing agent
        print(f"Using existing agent {agent}")
        agent_config = AgentConfig.load(agent)
        print("State path:", agent_config.save_state_dir())
        print("Persistent manager path:", agent_config.save_persistence_manager_dir())
        print("Index path:", agent_config.save_agent_index_dir())
        # persistence_manager = LocalStateManager(agent_config).load() # TODO: implement load
        # TODO: load prior agent state
        assert not any(
            [persona, human, model]
        ), f"Cannot override existing agent state with command line arguments: {persona}, {human}, {model}"

        # load existing agent
        memgpt_agent = AgentAsync.load_agent(memgpt.interface, agent_config)
    else:  # create new agent
        # create new agent config: override defaults with args if provided
        agent_config = AgentConfig(
            name=agent if agent else None,
            persona=persona if persona else config.default_persona,
            human=human if human else config.default_human,
            model=model if model else config.model,
        )

        # attach data source to agent
        agent_config.attach_data_source(data_source)

        # TODO: allow configrable state manager (only local is supported right now)
        persistence_manager = LocalStateManager(agent_config)  # TODO: insert dataset/pre-fill

        # save new agent config
        agent_config.save()
        print(f"Created new agent {agent_config.name}")

        # create agent
        memgpt_agent = presets.use_preset(
            presets.DEFAULT,
            agent_config,
            agent_config.model,
            agent_config.persona,
            agent_config.human,
            memgpt.interface,
            persistence_manager,
        )

    # start event loop
    from memgpt.main import run_agent_loop

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agent_loop(memgpt_agent, first, no_verify, config))  # TODO: add back no_verify
