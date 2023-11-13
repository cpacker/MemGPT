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
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.connectors.storage import StorageConnector
from memgpt.constants import LLM_MAX_TOKENS

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
        backend_options = ["webui", "llamacpp", "koboldcpp", "ollama", "lmstudio", "openai"]
        valid_config = config.model_endpoint_type in backend_options
        model_endpoint_type = questionary.select(
            "Select LLM backend (select 'openai' if you have an OpenAI compatible proxy):",
            backend_options,
            default=config.model_endpoint_type if valid_config else backend_options[0],
        ).ask()

        # set default endpoint
        default_model_endpoint = os.getenv("OPENAI_API_BASE")
        model_endpoint = questionary.text("Enter default endpoint:", default=default_model_endpoint).ask()
        assert model_endpoint, f"Enviornment variable OPENAI_API_BASE must be set."

    return model_endpoint_type, model_endpoint


def configure_model(config: MemGPTConfig, model_endpoint_type: str):
    # set: model, model_wrapper
    model, model_wrapper = None, None
    if model_endpoint_type == "openai" or model_endpoint_type == "azure":
        model_options = ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo-16k"]
        # TODO: select
        valid_model = config.model in model_options
        model = questionary.select(
            "Select default model (recommended: gpt-4):", choices=model_options, default=config.model if valid_model else model_options[0]
        ).ask()
    else:  # local models
        model_wrapper = questionary.text(
            "Enter default model wrapper:", default=config.model_wrapper if config.model_wrapper else "airoboros-l2-70b-2.1"
        ).ask()

        # ollama also needs model type
        if model_endpoint_type == "ollama":
            model = questionary.text("Enter default model type (e.g. airboros):", default=config.model if config.model else "").ask()
            model = None if len(model) == 0 else model

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
    from memgpt.presets import DEFAULT_PRESET, preset_options

    # configure preset
    default_preset = questionary.select("Select default preset:", preset_options, default=config.preset).ask()
    # defaults
    personas = [os.path.basename(f).replace(".txt", "") for f in utils.list_persona_files()]
    # print(personas)
    default_persona = questionary.select("Select default persona:", personas, default=config.default_persona).ask()
    humans = [os.path.basename(f).replace(".txt", "") for f in utils.list_human_files()]
    # print(humans)
    default_human = questionary.select("Select default human:", humans, default=config.default_human).ask()

    # TODO: figure out if we should set a default agent or not
    default_agent = None

    return default_preset, default_persona, default_human, default_agent


def configure_archival_storage(config: MemGPTConfig):
    # Configure archival storage backend
    archival_storage_options = ["local", "postgres"]
    archival_storage_type = questionary.select(
        "Select storage backend for archival data:", archival_storage_options, default=config.archival_storage_type
    ).ask()
    archival_storage_uri = None
    if archival_storage_type == "postgres":
        archival_storage_uri = questionary.text(
            "Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}):",
            default=config.archival_storage_uri if config.archival_storage_uri else "",
        ).ask()
    return archival_storage_type, archival_storage_uri


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
            raise ValueError("Missing enviornment variables for Azure. Please set then run `memgpt configure` again.")
    if model_endpoint_type == "openai" or embedding_endpoint_type == "openai":
        if not openai_key:
            raise ValueError("Missing enviornment variables for OpenAI. Please set them and run `memgpt configure` again.")

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
        default_persona=default_persona,
        default_human=default_human,
        default_agent=default_agent,
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
