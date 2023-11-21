import builtins
import questionary
import openai
from prettytable import PrettyTable
import typer
import os
import shutil
from collections import defaultdict

# from memgpt.cli import app
from memgpt import utils

import memgpt.humans.humans as humans
import memgpt.personas.personas as personas
from memgpt.config import MemGPTConfig, AgentConfig, Config
from memgpt.constants import MEMGPT_DIR
from memgpt.connectors.storage import StorageConnector
from memgpt.constants import LLM_MAX_TOKENS
from memgpt.local_llm.constants import DEFAULT_ENDPOINTS, DEFAULT_OLLAMA_MODEL, DEFAULT_WRAPPER_NAME
from memgpt.local_llm.utils import get_available_wrappers

app = typer.Typer()


def get_azure_credentials():
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    return azure_key, azure_endpoint, azure_version, azure_deployment, azure_embedding_deployment


def get_openai_credentials():
    openai_key = os.getenv("OPENAI_API_KEY")
    return openai_key


def configure_llm_endpoint(config: MemGPTConfig):
    # configure model endpoint
    model_endpoint_type, model_endpoint = None, None

    # get default
    default_model_endpoint_type = config.model_endpoint_type
    if config.model_endpoint_type is not None and config.model_endpoint_type not in ["openai", "azure"]:  # local model
        default_model_endpoint_type = "local"

    provider = questionary.select(
        "Select LLM inference provider:", choices=["openai", "azure", "local"], default=default_model_endpoint_type
    ).ask()

    # set: model_endpoint_type, model_endpoint
    if provider == "openai":
        model_endpoint_type = "openai"
        model_endpoint = "https://api.openai.com/v1"
        model_endpoint = questionary.text("Override default endpoint:", default=model_endpoint).ask()
        provider = "openai"
    elif provider == "azure":
        model_endpoint_type = "azure"
        _, model_endpoint, _, _, _ = get_azure_credentials()
    else:  # local models
        backend_options = ["webui", "webui-legacy", "llamacpp", "koboldcpp", "ollama", "lmstudio", "vllm", "openai"]
        default_model_endpoint_type = None
        if config.model_endpoint_type in backend_options:
            # set from previous config
            default_model_endpoint_type = config.model_endpoint_type
        else:
            # set form env variable (ok if none)
            default_model_endpoint_type = os.getenv("BACKEND_TYPE")
        model_endpoint_type = questionary.select(
            "Select LLM backend (select 'openai' if you have an OpenAI compatible proxy):",
            backend_options,
            default=default_model_endpoint_type,
        ).ask()

        # set default endpoint
        # if OPENAI_API_BASE is set, assume that this is the IP+port the user wanted to use
        default_model_endpoint = os.getenv("OPENAI_API_BASE")
        # if OPENAI_API_BASE is not set, try to pull a default IP+port format from a hardcoded set
        if default_model_endpoint is None:
            if model_endpoint_type in DEFAULT_ENDPOINTS:
                default_model_endpoint = DEFAULT_ENDPOINTS[model_endpoint_type]
                model_endpoint = questionary.text("Enter default endpoint:", default=default_model_endpoint).ask()
            else:
                # default_model_endpoint = None
                model_endpoint = None
                while not model_endpoint:
                    model_endpoint = questionary.text("Enter default endpoint:").ask()
                    if "http://" not in model_endpoint and "https://" not in model_endpoint:
                        typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                        model_endpoint = None
        else:
            model_endpoint = default_model_endpoint
        assert model_endpoint, f"Environment variable OPENAI_API_BASE must be set."

    return model_endpoint_type, model_endpoint


def configure_model(config: MemGPTConfig, model_endpoint_type: str):
    # set: model, model_wrapper
    model, model_wrapper = None, None
    if model_endpoint_type == "openai" or model_endpoint_type == "azure":
        model_options = ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        # TODO: select
        valid_model = config.model in model_options
        model = questionary.select(
            "Select default model (recommended: gpt-4):", choices=model_options, default=config.model if valid_model else model_options[0]
        ).ask()
    else:  # local models
        # ollama also needs model type
        if model_endpoint_type == "ollama":
            default_model = config.model if config.model and config.model_endpoint_type == "ollama" else DEFAULT_OLLAMA_MODEL
            model = questionary.text(
                "Enter default model name (required for Ollama, see: https://memgpt.readthedocs.io/en/latest/ollama):",
                default=default_model,
            ).ask()
            model = None if len(model) == 0 else model

        # vllm needs huggingface model tag
        if model_endpoint_type == "vllm":
            default_model = config.model if config.model and config.model_endpoint_type == "vllm" else ""
            model = questionary.text(
                "Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b):",
                default=default_model,
            ).ask()
            model = None if len(model) == 0 else model
            model_wrapper = None  # no model wrapper for vLLM

        # model wrapper
        if model_endpoint_type != "vllm":
            available_model_wrappers = builtins.list(get_available_wrappers().keys())
            model_wrapper = questionary.select(
                f"Select default model wrapper (recommended: {DEFAULT_WRAPPER_NAME}):",
                choices=available_model_wrappers,
                default=DEFAULT_WRAPPER_NAME,
            ).ask()

    # set: context_window
    if str(model) not in LLM_MAX_TOKENS:
        # Ask the user to specify the context length
        context_length_options = [
            str(2**12),  # 4096
            str(2**13),  # 8192
            str(2**14),  # 16384
            str(2**15),  # 32768
            str(2**18),  # 262144
            "custom",  # enter yourself
        ]
        context_window = questionary.select(
            "Select your model's context window (for Mistral 7B models, this is probably 8k / 8192):",
            choices=context_length_options,
            default=str(LLM_MAX_TOKENS["DEFAULT"]),
        ).ask()

        # If custom, ask for input
        if context_window == "custom":
            while True:
                context_window = questionary.text("Enter context window (e.g. 8192)").ask()
                try:
                    context_window = int(context_window)
                    break
                except ValueError:
                    print(f"Context window must be a valid integer")
        else:
            context_window = int(context_window)
    else:
        # Pull the context length from the models
        context_window = LLM_MAX_TOKENS[model]
    return model, model_wrapper, context_window


def configure_embedding_endpoint(config: MemGPTConfig):
    # configure embedding endpoint

    default_embedding_endpoint_type = config.embedding_endpoint_type
    if config.embedding_endpoint_type is not None and config.embedding_endpoint_type not in ["openai", "azure"]:  # local model
        default_embedding_endpoint_type = "local"

    embedding_endpoint_type, embedding_endpoint, embedding_dim = None, None, None
    embedding_provider = questionary.select(
        "Select embedding provider:", choices=["openai", "azure", "local"], default=default_embedding_endpoint_type
    ).ask()
    if embedding_provider == "openai":
        embedding_endpoint_type = "openai"
        embedding_endpoint = "https://api.openai.com/v1"
        embedding_dim = 1536
    elif embedding_provider == "azure":
        embedding_endpoint_type = "azure"
        _, _, _, _, embedding_endpoint = get_azure_credentials()
        embedding_dim = 1536
    else:  # local models
        embedding_endpoint_type = "local"
        embedding_endpoint = None
        embedding_dim = 384
    return embedding_endpoint_type, embedding_endpoint, embedding_dim


def configure_cli(config: MemGPTConfig):
    # set: preset, default_persona, default_human, default_agent``
    from memgpt.presets.presets import preset_options

    # preset
    default_preset = config.preset if config.preset and config.preset in preset_options else None
    preset = questionary.select("Select default preset:", preset_options, default=default_preset).ask()

    # persona
    personas = [os.path.basename(f).replace(".txt", "") for f in utils.list_persona_files()]
    default_persona = config.persona if config.persona and config.persona in personas else None
    persona = questionary.select("Select default persona:", personas, default=default_persona).ask()

    # human
    humans = [os.path.basename(f).replace(".txt", "") for f in utils.list_human_files()]
    default_human = config.human if config.human and config.human in humans else None
    human = questionary.select("Select default human:", humans, default=default_human).ask()

    # TODO: figure out if we should set a default agent or not
    agent = None

    return preset, persona, human, agent


def configure_archival_storage(config: MemGPTConfig):
    # Configure archival storage backend
    archival_storage_options = ["local", "lancedb", "postgres"]
    archival_storage_type = questionary.select(
        "Select storage backend for archival data:", archival_storage_options, default=config.archival_storage_type
    ).ask()
    archival_storage_uri = None
    if archival_storage_type == "postgres":
        archival_storage_uri = questionary.text(
            "Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}):",
            default=config.archival_storage_uri if config.archival_storage_uri else "",
        ).ask()

    if archival_storage_type == "lancedb":
        archival_storage_uri = questionary.text(
            "Enter lanncedb connection string (e.g. ./.lancedb",
            default=config.archival_storage_uri if config.archival_storage_uri else "./.lancedb",
        ).ask()

    return archival_storage_type, archival_storage_uri

    # TODO: allow configuring embedding model


@app.command()
def configure():
    """Updates default MemGPT configurations"""

    MemGPTConfig.create_config_dir()

    # Will pre-populate with defaults, or what the user previously set
    config = MemGPTConfig.load()
    model_endpoint_type, model_endpoint = configure_llm_endpoint(config)
    model, model_wrapper, context_window = configure_model(config, model_endpoint_type)
    embedding_endpoint_type, embedding_endpoint, embedding_dim = configure_embedding_endpoint(config)
    default_preset, default_persona, default_human, default_agent = configure_cli(config)
    archival_storage_type, archival_storage_uri = configure_archival_storage(config)

    # check credentials
    azure_key, azure_endpoint, azure_version, azure_deployment, azure_embedding_deployment = get_azure_credentials()
    openai_key = get_openai_credentials()
    if model_endpoint_type == "azure" or embedding_endpoint_type == "azure":
        if all([azure_key, azure_endpoint, azure_version]):
            print(f"Using Microsoft endpoint {azure_endpoint}.")
            if all([azure_deployment, azure_embedding_deployment]):
                print(f"Using deployment id {azure_deployment}")
        else:
            raise ValueError(
                "Missing environment variables for Azure (see https://memgpt.readthedocs.io/en/latest/endpoints/#azure). Please set then run `memgpt configure` again."
            )
    if model_endpoint_type == "openai" or embedding_endpoint_type == "openai":
        if not openai_key:
            raise ValueError(
                "Missing environment variables for OpenAI (see https://memgpt.readthedocs.io/en/latest/endpoints/#openai). Please set them and run `memgpt configure` again."
            )

    config = MemGPTConfig(
        # model configs
        model=model,
        model_endpoint=model_endpoint,
        model_endpoint_type=model_endpoint_type,
        model_wrapper=model_wrapper,
        context_window=context_window,
        # embedding configs
        embedding_endpoint_type=embedding_endpoint_type,
        embedding_endpoint=embedding_endpoint,
        embedding_dim=embedding_dim,
        # cli configs
        preset=default_preset,
        persona=default_persona,
        human=default_human,
        agent=default_agent,
        # credentials
        openai_key=openai_key,
        azure_key=azure_key,
        azure_endpoint=azure_endpoint,
        azure_version=azure_version,
        azure_deployment=azure_deployment,
        azure_embedding_deployment=azure_embedding_deployment,
        # storage
        archival_storage_type=archival_storage_type,
        archival_storage_uri=archival_storage_uri,
    )
    print(f"Saving config to {config.config_path}")
    config.save()


@app.command()
def list(option: str):
    if option == "agents":
        """List all agents"""
        table = PrettyTable()
        table.field_names = ["Name", "Model", "Persona", "Human", "Data Source", "Create Time"]
        for agent_file in utils.list_agent_config_files():
            agent_name = os.path.basename(agent_file).replace(".json", "")
            agent_config = AgentConfig.load(agent_name)
            table.add_row(
                [
                    agent_name,
                    agent_config.model,
                    agent_config.persona,
                    agent_config.human,
                    ",".join(agent_config.data_sources),
                    agent_config.create_time,
                ]
            )
        print(table)
    elif option == "humans":
        """List all humans"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for human_file in utils.list_human_files():
            text = open(human_file, "r").read()
            name = os.path.basename(human_file).replace("txt", "")
            table.add_row([name, text])
        print(table)
    elif option == "personas":
        """List all personas"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for persona_file in utils.list_persona_files():
            print(persona_file)
            text = open(persona_file, "r").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            table.add_row([name, text])
        print(table)
    elif option == "sources":
        """List all data sources"""
        table = PrettyTable()
        table.field_names = ["Name", "Location", "Agents"]
        config = MemGPTConfig.load()
        # TODO: eventually look accross all storage connections
        # TODO: add data source stats
        source_to_agents = {}
        for agent_file in utils.list_agent_config_files():
            agent_name = os.path.basename(agent_file).replace(".json", "")
            agent_config = AgentConfig.load(agent_name)
            for ds in agent_config.data_sources:
                if ds in source_to_agents:
                    source_to_agents[ds].append(agent_name)
                else:
                    source_to_agents[ds] = [agent_name]
        for data_source in StorageConnector.list_loaded_data():
            location = config.archival_storage_type
            agents = ",".join(source_to_agents[data_source]) if data_source in source_to_agents else ""
            table.add_row([data_source, location, agents])
        print(table)
    else:
        raise ValueError(f"Unknown option {option}")


@app.command()
def add(
    option: str,  # [human, persona]
    name: str = typer.Option(help="Name of human/persona"),
    text: str = typer.Option(None, help="Text of human/persona"),
    filename: str = typer.Option(None, "-f", help="Specify filename"),
):
    """Add a person/human"""

    if option == "persona":
        directory = os.path.join(MEMGPT_DIR, "personas")
    elif option == "human":
        directory = os.path.join(MEMGPT_DIR, "humans")
    else:
        raise ValueError(f"Unknown kind {kind}")

    if filename:
        assert text is None, f"Cannot provide both filename and text"
        # copy file to directory
        shutil.copyfile(filename, os.path.join(directory, name))
    if text:
        assert filename is None, f"Cannot provide both filename and text"
        # write text to file
        with open(os.path.join(directory, name), "w") as f:
            f.write(text)
