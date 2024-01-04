import os
import shutil
import typer

from memgpt.agent import Agent
from memgpt.presets.presets import use_preset
from memgpt.persistence_manager import LocalStateManager
from memgpt.config import AgentConfig
from memgpt.utils import printd, suppress_stdout, get_human_text, get_persona_text
from memgpt.constants import CLI_WARNING_PREFIX, MEMGPT_DIR
from memgpt.interface import CLIInterface as interface  # for printing to terminal


def load_agent(
    persona: str = None,
    agent: str = None,
    human: str = None,
    # model flags
    model: str = None,
    model_wrapper: str = None,
    model_endpoint: str = None,
    model_endpoint_type: str = None,
    context_window: int = None,
) -> Agent:
    """Load an existing agent"""
    agent_config = AgentConfig.load(agent)
    printd("State path:", agent_config.save_state_dir())
    printd("Persistent manager path:", agent_config.save_persistence_manager_dir())
    printd("Index path:", agent_config.save_agent_index_dir())

    # persistence_manager = LocalStateManager(agent_config).load() # TODO: implement load
    # TODO: load prior agent state
    if persona and persona != agent_config.persona:
        typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing persona {agent_config.persona} with {persona}", fg=typer.colors.YELLOW)
        agent_config.persona = persona
        # raise ValueError(f"Cannot override {agent_config.name} existing persona {agent_config.persona} with {persona}")
    if human and human != agent_config.human:
        typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing human {agent_config.human} with {human}", fg=typer.colors.YELLOW)
        agent_config.human = human
        # raise ValueError(f"Cannot override {agent_config.name} existing human {agent_config.human} with {human}")

    # Allow overriding model specifics (model, model wrapper, model endpoint IP + type, context_window)
    if model and model != agent_config.model:
        typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing model {agent_config.model} with {model}", fg=typer.colors.YELLOW)
        agent_config.model = model
    if context_window is not None and int(context_window) != agent_config.context_window:
        typer.secho(
            f"{CLI_WARNING_PREFIX}Overriding existing context window {agent_config.context_window} with {context_window}",
            fg=typer.colors.YELLOW,
        )
        agent_config.context_window = context_window
    if model_wrapper and model_wrapper != agent_config.model_wrapper:
        typer.secho(
            f"{CLI_WARNING_PREFIX}Overriding existing model wrapper {agent_config.model_wrapper} with {model_wrapper}",
            fg=typer.colors.YELLOW,
        )
        agent_config.model_wrapper = model_wrapper
    if model_endpoint and model_endpoint != agent_config.model_endpoint:
        typer.secho(
            f"{CLI_WARNING_PREFIX}Overriding existing model endpoint {agent_config.model_endpoint} with {model_endpoint}",
            fg=typer.colors.YELLOW,
        )
        agent_config.model_endpoint = model_endpoint
    if model_endpoint_type and model_endpoint_type != agent_config.model_endpoint_type:
        typer.secho(
            f"{CLI_WARNING_PREFIX}Overriding existing model endpoint type {agent_config.model_endpoint_type} with {model_endpoint_type}",
            fg=typer.colors.YELLOW,
        )
        agent_config.model_endpoint_type = model_endpoint_type

    # Update the agent config with any overrides
    agent_config.save()

    # Supress llama-index noise
    with suppress_stdout():
        # load existing agent
        memgpt_agent = Agent.load_agent(interface, agent_config)

    return memgpt_agent


def create_agent(
    persona: str = None,
    agent: str = None,
    human: str = None,
    preset: str = None,
    # model flags
    model: str = None,
    model_wrapper: str = None,
    model_endpoint: str = None,
    model_endpoint_type: str = None,
    context_window: int = None,
) -> Agent:
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

    # save new agent config
    agent_config.save()

    # Supress llama-index noise
    with suppress_stdout():
        # TODO: allow configrable state manager (only local is supported right now)
        persistence_manager = LocalStateManager(agent_config)  # TODO: insert dataset/pre-fill

    # create agent
    try:
        memgpt_agent = use_preset(
            agent_config.preset,
            agent_config,
            agent_config.model,
            get_persona_text(agent_config.persona),
            get_human_text(agent_config.human),
            interface,
            persistence_manager,
        )
    except ValueError as e:
        typer.secho(f"Failed to create agent from provided information:\n{e}", fg=typer.colors.RED)
        # Delete the directory of the failed agent
        try:
            # Path to the specific file
            agent_config_file = agent_config.agent_config_path

            # Check if the file exists
            if os.path.isfile(agent_config_file):
                # Delete the file
                os.remove(agent_config_file)

            # Now, delete the directory along with any remaining files in it
            agent_save_dir = os.path.join(MEMGPT_DIR, "agents", agent_config.name)
            shutil.rmtree(agent_save_dir)
        except:
            typer.secho(f"Failed to delete agent directory during cleanup:\n{e}", fg=typer.colors.RED)
            raise
        raise

    return memgpt_agent
