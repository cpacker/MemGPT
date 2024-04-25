import uuid
import sys
import logging
from pathlib import Path
import os
import subprocess
from enum import Enum
from typing import Annotated, Optional

import typer
import questionary

from memgpt.log import logger
from memgpt.interface import CLIInterface as interface  # for printing to terminal
import memgpt.utils as utils
from memgpt.utils import printd
from memgpt.config import MemGPTConfig
from memgpt.constants import CLI_WARNING_PREFIX
from memgpt.agent import Agent, save_agent
from memgpt.server.constants import WS_DEFAULT_PORT, REST_DEFAULT_PORT
from memgpt.data_types import User
from memgpt.metadata import MetadataStore


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


def generate_self_signed_cert(cert_path="selfsigned.crt", key_path="selfsigned.key"):
    """Generate a self-signed SSL certificate.

    NOTE: intended to be used for development only.
    """
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:4096",
            "-keyout",
            key_path,
            "-out",
            cert_path,
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/C=US/ST=Denial/L=Springfield/O=Dis/CN=localhost",
        ],
        check=True,
    )
    return cert_path, key_path


def server(
    type: Annotated[ServerChoice, typer.Option(help="Server to run")] = "rest",
    port: Annotated[Optional[int], typer.Option(help="Port to run the server on")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host to run the server on (default to localhost)")] = None,
    use_ssl: Annotated[bool, typer.Option(help="Run the server using HTTPS?")] = False,
    ssl_cert: Annotated[Optional[str], typer.Option(help="Path to SSL certificate (if use_ssl is True)")] = None,
    ssl_key: Annotated[Optional[str], typer.Option(help="Path to SSL key file (if use_ssl is True)")] = None,
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
            if use_ssl:
                if ssl_cert is None:  # No certificate path provided, generate a self-signed certificate
                    ssl_certfile, ssl_keyfile = generate_self_signed_cert()
                    print(f"Running server with self-signed SSL cert: {ssl_certfile}, {ssl_keyfile}")
                else:
                    ssl_certfile, ssl_keyfile = ssl_cert, ssl_key  # Assuming cert includes both
                    print(f"Running server with provided SSL cert: {ssl_certfile}, {ssl_keyfile}")

                # This will start the server on HTTPS
                assert isinstance(ssl_certfile, str) and os.path.exists(ssl_certfile), ssl_certfile
                assert isinstance(ssl_keyfile, str) and os.path.exists(ssl_keyfile), ssl_keyfile
                print(
                    f"Running: uvicorn {app}:app --host {host or 'localhost'} --port {port or REST_DEFAULT_PORT} --ssl-keyfile {ssl_keyfile} --ssl-certfile {ssl_certfile}"
                )
                uvicorn.run(
                    app,
                    host=host or "localhost",
                    port=port or REST_DEFAULT_PORT,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                )
            else:
                # Start the subprocess in a new session
                print(f"Running: uvicorn {app}:app --host {host or 'localhost'} --port {port or REST_DEFAULT_PORT}")
                uvicorn.run(
                    app,
                    host=host or "localhost",
                    port=port or REST_DEFAULT_PORT,
                )

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

    if not MemGPTConfig.exists():
        raise Exception("No MemGPT config found")

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
        memgpt_agent = Agent(agent_state=agent_state, interface=interface())

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
        try:
            preset_obj = ms.get_preset(preset_name=preset if preset else config.preset, user_id=user.id)
            if preset_obj is None:
                typer.secho("Couldn't find presets in database, please run `add to db`", fg=typer.colors.RED)
                sys.exit(1)

            # Overwrite fields in the preset if they were specified
            preset_obj.human = human if human else config.human
            preset_obj.persona = persona if persona else config.persona

            typer.secho(f"->  🤖 Using persona profile '{preset_obj.persona}'", fg=typer.colors.WHITE)
            typer.secho(f"->  🧑 Using human profile '{preset_obj.human}'", fg=typer.colors.WHITE)

            memgpt_agent = Agent(
                interface=interface(),
                name=agent_name,
                created_by=user.id,
                preset=preset_obj,
                llm_config=llm_config,
                embedding_config=embedding_config,
                # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
                first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
            )
            save_agent(agent=memgpt_agent, ms=ms)

        except ValueError as e:
            typer.secho(f"Failed to create agent from provided information:\n{e}", fg=typer.colors.RED)
            sys.exit(1)
        typer.secho(f"🎉 Created new agent '{memgpt_agent.agent_state.name}' (id={memgpt_agent.agent_state.id})", fg=typer.colors.GREEN)

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
    except Exception:
        typer.secho(f"Failed to delete agent '{agent_name}' (id={agent.id})", fg=typer.colors.RED)
        sys.exit(1)


def version():
    import memgpt

    print(memgpt.__version__)
    return memgpt.__version__
