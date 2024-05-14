import uuid
import sys
import logging
from typing import Annotated, Optional

import typer

from memgpt.log import logger
from memgpt.interface import CLIInterface as interface  # for printing to terminal
import memgpt.utils as utils
from memgpt.utils import printd
from memgpt.config import MemGPTConfig
from memgpt.constants import CLI_WARNING_PREFIX
from memgpt.agent import Agent, save_agent
from memgpt.data_types import Preset, User
from memgpt.metadata import MetadataStore


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


def run(
    persona: Annotated[Optional[str], typer.Option(help="Specify persona")] = None,
    agent: Annotated[Optional[str], typer.Option(help="Specify agent name")] = None,
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
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
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

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    config = MemGPTConfig.load()

    # read user id from config
    ms = MetadataStore(config)
    user = create_default_user_or_exit(config, ms)

    # determine agent to use, if not provided
    if not agent:
        raise Exception("Please provide an agent name")

    # create agent config
    agent_state = ms.get_agent(agent_name=agent, user_id=user.id) if agent else None
    if agent and agent_state:  # use existing agent
        typer.secho(f"\n🔁 Using existing agent {agent}", fg=typer.colors.GREEN)
        printd("Loading agent state:", agent_state.id)
        printd("Agent state:", agent_state.state)
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
        memgpt_agent = Agent(agent_state=agent_state, interface=interface())

    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)
        if not agent:
            raise Exception("Specify an agent name")

        agent_name = agent
        llm_config = config.default_llm_config
        embedding_config = config.default_embedding_config  # TODO allow overriding embedding params via CLI run

        # create agent
        try:
            preset_obj = ms.get_preset(preset_name=preset if preset else config.preset, user_id=user.id)
            if preset_obj is None:
                preset_obj = Preset(description="foo", system="you are a robot", human="you are a human", persona="you are a persona", name="test", user_id=uuid.uuid4())  # type: ignore

            # Overwrite fields in the preset if they were specified
            preset_obj.human = human if human else config.human
            typer.secho(f"->  🧑 Using human profile '{preset_obj.human}'", fg=typer.colors.WHITE)

            memgpt_agent = Agent(
                interface=interface(),
                name=agent_name,
                created_by=user.id,
                preset=preset_obj,
                llm_config=llm_config,
                embedding_config=embedding_config,
            )
            save_agent(agent=memgpt_agent, ms=ms)

        except ValueError as e:
            typer.secho(f"Failed to create agent from provided information:\n{e}", fg=typer.colors.RED)
            sys.exit(1)
        typer.secho(f"🎉 Created new agent '{memgpt_agent.agent_state.name}' (id={memgpt_agent.agent_state.id})", fg=typer.colors.GREEN)

    # start event loop
    from memgpt.main import run_agent_loop

    print()  # extra space
    run_agent_loop(
        memgpt_agent,
        config=config,
        ms=ms,
    )
