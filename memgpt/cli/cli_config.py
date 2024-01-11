import builtins
import uuid
import questionary
from prettytable import PrettyTable
import typer
import os
import shutil
from typing import Annotated
from enum import Enum

# from memgpt.cli import app
from memgpt import utils

from memgpt.config import MemGPTConfig
from memgpt.constants import MEMGPT_DIR

# from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.constants import LLM_MAX_TOKENS
from memgpt.local_llm.constants import DEFAULT_ENDPOINTS, DEFAULT_OLLAMA_MODEL, DEFAULT_WRAPPER_NAME
from memgpt.local_llm.utils import get_available_wrappers
from memgpt.openai_tools import openai_get_model_list, azure_openai_get_model_list, smart_urljoin
from memgpt.server.utils import shorten_key_middle
from memgpt.data_types import User, LLMConfig, EmbeddingConfig
from memgpt.metadata import MetadataStore
from memgpt.agent_store.storage import StorageConnector, TableType

app = typer.Typer()


def get_azure_credentials():
    creds = dict(
        azure_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )
    # embedding endpoint and version default to non-embedding
    creds["azure_embedding_endpoint"] = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", creds["azure_endpoint"])
    creds["azure_embedding_version"] = os.getenv("AZURE_OPENAI_EMBEDDING_VERSION", creds["azure_version"])
    return creds


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
        # check for key
        if config.openai_key is None:
            # allow key to get pulled from env vars
            openai_api_key = os.getenv("OPENAI_API_KEY", None)
            if openai_api_key is None:
                # if we still can't find it, ask for it as input
                while openai_api_key is None or len(openai_api_key) == 0:
                    # Ask for API key as input
                    openai_api_key = questionary.text(
                        "Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):"
                    ).ask()
            config.openai_key = openai_api_key
            config.save()
        else:
            # Give the user an opportunity to overwrite the key
            openai_api_key = None
            default_input = shorten_key_middle(config.openai_key) if config.openai_key.startswith("sk-") else config.openai_key
            openai_api_key = questionary.text(
                "Enter your OpenAI API key (hit enter to use existing key):",
                default=default_input,
            ).ask()
            # If the user modified it, use the new one
            if openai_api_key != default_input:
                config.openai_key = openai_api_key
                config.save()

        model_endpoint_type = "openai"
        model_endpoint = "https://api.openai.com/v1"
        model_endpoint = questionary.text("Override default endpoint:", default=model_endpoint).ask()
        provider = "openai"

    elif provider == "azure":
        # check for necessary vars
        azure_creds = get_azure_credentials()
        if not all([azure_creds["azure_key"], azure_creds["azure_endpoint"], azure_creds["azure_version"]]):
            raise ValueError(
                "Missing environment variables for Azure (see https://memgpt.readme.io/docs/endpoints#azure-openai). Please set then run `memgpt configure` again."
            )
        else:
            config.azure_key = azure_creds["azure_key"]
            config.azure_endpoint = azure_creds["azure_endpoint"]
            config.azure_version = azure_creds["azure_version"]
            config.save()

        model_endpoint_type = "azure"
        model_endpoint = azure_creds["azure_endpoint"]

    else:  # local models
        backend_options = ["webui", "webui-legacy", "llamacpp", "koboldcpp", "ollama", "lmstudio", "lmstudio-legacy", "vllm", "openai"]
        default_model_endpoint_type = None
        if config.model_endpoint_type in backend_options:
            # set from previous config
            default_model_endpoint_type = config.model_endpoint_type
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
                while not utils.is_valid_url(model_endpoint):
                    typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                    model_endpoint = questionary.text("Enter default endpoint:", default=default_model_endpoint).ask()
            elif config.model_endpoint:
                model_endpoint = questionary.text("Enter default endpoint:", default=config.model_endpoint).ask()
                while not utils.is_valid_url(model_endpoint):
                    typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                    model_endpoint = questionary.text("Enter default endpoint:", default=config.model_endpoint).ask()
            else:
                # default_model_endpoint = None
                model_endpoint = None
                model_endpoint = questionary.text("Enter default endpoint:").ask()
                while not utils.is_valid_url(model_endpoint):
                    typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                    model_endpoint = questionary.text("Enter default endpoint:").ask()
        else:
            model_endpoint = default_model_endpoint
        assert model_endpoint, f"Environment variable OPENAI_API_BASE must be set."

    return model_endpoint_type, model_endpoint


def configure_model(config: MemGPTConfig, model_endpoint_type: str, model_endpoint: str):
    # set: model, model_wrapper
    model, model_wrapper = None, None
    if model_endpoint_type == "openai" or model_endpoint_type == "azure":
        # Get the model list from the openai / azure endpoint
        hardcoded_model_options = ["gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        fetched_model_options = None
        try:
            if model_endpoint_type == "openai":
                fetched_model_options = openai_get_model_list(url=model_endpoint, api_key=config.openai_key)
            elif model_endpoint_type == "azure":
                fetched_model_options = azure_openai_get_model_list(
                    url=model_endpoint, api_key=config.azure_key, api_version=config.azure_version
                )
            fetched_model_options = [obj["id"] for obj in fetched_model_options["data"] if obj["id"].startswith("gpt-")]
        except:
            # NOTE: if this fails, it means the user's key is probably bad
            typer.secho(
                f"Failed to get model list from {model_endpoint} - make sure your API key and endpoints are correct!", fg=typer.colors.RED
            )

        # First ask if the user wants to see the full model list (some may be incompatible)
        see_all_option_str = "[see all options]"
        other_option_str = "[enter model name manually]"

        # Check if the model we have set already is even in the list (informs our default)
        valid_model = config.model in hardcoded_model_options
        model = questionary.select(
            "Select default model (recommended: gpt-4):",
            choices=hardcoded_model_options + [see_all_option_str, other_option_str],
            default=config.model if valid_model else hardcoded_model_options[0],
        ).ask()

        # If the user asked for the full list, show it
        if model == see_all_option_str:
            typer.secho(f"Warning: not all models shown are guaranteed to work with MemGPT", fg=typer.colors.RED)
            model = questionary.select(
                "Select default model (recommended: gpt-4):",
                choices=fetched_model_options + [other_option_str],
                default=config.model if valid_model else fetched_model_options[0],
            ).ask()

        # Finally if the user asked to manually input, allow it
        if model == other_option_str:
            model = ""
            while len(model) == 0:
                model = questionary.text(
                    "Enter custom model name:",
                ).ask()

    else:  # local models
        # ollama also needs model type
        if model_endpoint_type == "ollama":
            default_model = config.model if config.model and config.model_endpoint_type == "ollama" else DEFAULT_OLLAMA_MODEL
            model = questionary.text(
                "Enter default model name (required for Ollama, see: https://memgpt.readme.io/docs/ollama):",
                default=default_model,
            ).ask()
            model = None if len(model) == 0 else model

        default_model = config.model if config.model and config.model_endpoint_type == "vllm" else ""

        # vllm needs huggingface model tag
        if model_endpoint_type == "vllm":
            try:
                # Don't filter model list for vLLM since model list is likely much smaller than OpenAI/Azure endpoint
                # + probably has custom model names
                model_options = openai_get_model_list(url=smart_urljoin(model_endpoint, "v1"), api_key=None)
                model_options = [obj["id"] for obj in model_options["data"]]
            except:
                print(f"Failed to get model list from {model_endpoint}, using defaults")
                model_options = None

            # If we got model options from vLLM endpoint, allow selection + custom input
            if model_options is not None:
                other_option_str = "other (enter name)"
                valid_model = config.model in model_options
                model_options.append(other_option_str)
                model = questionary.select(
                    "Select default model:", choices=model_options, default=config.model if valid_model else model_options[0]
                ).ask()

                # If we got custom input, ask for raw input
                if model == other_option_str:
                    model = questionary.text(
                        "Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b):",
                        default=default_model,
                    ).ask()
                    # TODO allow empty string for input?
                    model = None if len(model) == 0 else model

            else:
                model = questionary.text(
                    "Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b):",
                    default=default_model,
                ).ask()
                model = None if len(model) == 0 else model

        # model wrapper
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

    embedding_endpoint_type, embedding_endpoint, embedding_dim, embedding_model = None, None, None, None
    embedding_provider = questionary.select(
        "Select embedding provider:", choices=["openai", "azure", "hugging-face", "local"], default=default_embedding_endpoint_type
    ).ask()

    if embedding_provider == "openai":
        # check for key
        if config.openai_key is None:
            # allow key to get pulled from env vars
            openai_api_key = os.getenv("OPENAI_API_KEY", None)
            if openai_api_key is None:
                # if we still can't find it, ask for it as input
                while openai_api_key is None or len(openai_api_key) == 0:
                    # Ask for API key as input
                    openai_api_key = questionary.text(
                        "Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):"
                    ).ask()
                config.openai_key = openai_api_key
                config.save()

        embedding_endpoint_type = "openai"
        embedding_endpoint = "https://api.openai.com/v1"
        embedding_dim = 1536

    elif embedding_provider == "azure":
        # check for necessary vars
        azure_creds = get_azure_credentials()
        if not all([azure_creds["azure_key"], azure_creds["azure_embedding_endpoint"], azure_creds["azure_embedding_version"]]):
            raise ValueError(
                "Missing environment variables for Azure (see https://memgpt.readme.io/docs/endpoints#azure-openai). Please set then run `memgpt configure` again."
            )
        # TODO we need to write these out to the config once we use them if we plan to ping for embedding lists with them

        embedding_endpoint_type = "azure"
        embedding_endpoint = azure_creds["azure_embedding_endpoint"]
        embedding_dim = 1536

    elif embedding_provider == "hugging-face":
        # configure hugging face embedding endpoint (https://github.com/huggingface/text-embeddings-inference)
        # supports custom model/endpoints
        embedding_endpoint_type = "hugging-face"
        embedding_endpoint = None

        # get endpoint
        embedding_endpoint = questionary.text("Enter default endpoint:").ask()
        while not utils.is_valid_url(embedding_endpoint):
            typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
            embedding_endpoint = questionary.text("Enter default endpoint:").ask()

        # get model type
        default_embedding_model = config.embedding_model if config.embedding_model else "BAAI/bge-large-en-v1.5"
        embedding_model = questionary.text(
            "Enter HuggingFace model tag (e.g. BAAI/bge-large-en-v1.5):",
            default=default_embedding_model,
        ).ask()

        # get model dimentions
        default_embedding_dim = config.embedding_dim if config.embedding_dim else "1024"
        embedding_dim = questionary.text("Enter embedding model dimentions (e.g. 1024):", default=str(default_embedding_dim)).ask()
        try:
            embedding_dim = int(embedding_dim)
        except Exception as e:
            raise ValueError(f"Failed to cast {embedding_dim} to integer.")
    else:  # local models
        embedding_endpoint_type = "local"
        embedding_endpoint = None
        embedding_dim = 384

    return embedding_endpoint_type, embedding_endpoint, embedding_dim, embedding_model


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
    archival_storage_options = ["postgres", "chroma"]
    archival_storage_type = questionary.select(
        "Select storage backend for archival data:", archival_storage_options, default=config.archival_storage_type
    ).ask()
    archival_storage_uri, archival_storage_path = config.archival_storage_uri, config.archival_storage_path

    # configure postgres
    if archival_storage_type == "postgres":
        archival_storage_uri = questionary.text(
            "Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}):",
            default=config.archival_storage_uri if config.archival_storage_uri else "",
        ).ask()

    # TODO: add back
    ## configure lancedb
    # if archival_storage_type == "lancedb":
    #    archival_storage_uri = questionary.text(
    #        "Enter lanncedb connection string (e.g. ./.lancedb",
    #        default=config.archival_storage_uri if config.archival_storage_uri else "./.lancedb",
    #    ).ask()

    # configure chroma
    if archival_storage_type == "chroma":
        chroma_type = questionary.select("Select chroma backend:", ["http", "persistent"], default="persistent").ask()
        if chroma_type == "http":
            archival_storage_uri = questionary.text("Enter chroma ip (e.g. localhost:8000):", default="localhost:8000").ask()
        if chroma_type == "persistent":
            archival_storage_path = os.path.join(MEMGPT_DIR, "chroma")

    return archival_storage_type, archival_storage_uri, archival_storage_path

    # TODO: allow configuring embedding model


def configure_recall_storage(config: MemGPTConfig):
    # Configure recall storage backend
    recall_storage_options = ["sqlite", "postgres"]
    recall_storage_type = questionary.select(
        "Select storage backend for recall data:", recall_storage_options, default=config.recall_storage_type
    ).ask()
    recall_storage_uri, recall_storage_path = config.recall_storage_uri, config.recall_storage_path
    # configure postgres
    if recall_storage_type == "postgres":
        recall_storage_uri = questionary.text(
            "Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}):",
            default=config.recall_storage_uri if config.recall_storage_uri else "",
        ).ask()

    return recall_storage_type, recall_storage_uri, recall_storage_path


@app.command()
def configure():
    """Updates default MemGPT configurations"""

    # check credentials
    openai_key = get_openai_credentials()
    azure_creds = get_azure_credentials()

    MemGPTConfig.create_config_dir()

    # Will pre-populate with defaults, or what the user previously set
    config = MemGPTConfig.load()
    try:
        model_endpoint_type, model_endpoint = configure_llm_endpoint(config)
        model, model_wrapper, context_window = configure_model(
            config=config, model_endpoint_type=model_endpoint_type, model_endpoint=model_endpoint
        )
        embedding_endpoint_type, embedding_endpoint, embedding_dim, embedding_model = configure_embedding_endpoint(config)
        default_preset, default_persona, default_human, default_agent = configure_cli(config)
        archival_storage_type, archival_storage_uri, archival_storage_path = configure_archival_storage(config)
        recall_storage_type, recall_storage_uri, recall_storage_path = configure_recall_storage(config)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        return

    # TODO: remove most of this (deplicated with User table)
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
        embedding_model=embedding_model,
        # cli configs
        preset=default_preset,
        persona=default_persona,
        human=default_human,
        agent=default_agent,
        # credentials
        openai_key=openai_key,
        azure_key=azure_creds["azure_key"],
        azure_endpoint=azure_creds["azure_endpoint"],
        azure_version=azure_creds["azure_version"],
        azure_deployment=azure_creds["azure_deployment"],  # OK if None
        azure_embedding_deployment=azure_creds["azure_embedding_deployment"],  # OK if None
        # storage
        archival_storage_type=archival_storage_type,
        archival_storage_uri=archival_storage_uri,
        archival_storage_path=archival_storage_path,
        # recall storage
        recall_storage_type=recall_storage_type,
        recall_storage_uri=recall_storage_uri,
        recall_storage_path=recall_storage_path,
        # metadata storage (currently forced to match recall storage)
        metadata_storage_type=recall_storage_type,
        metadata_storage_uri=recall_storage_uri,
        metadata_storage_path=recall_storage_path,
    )

    typer.secho(f"📖 Saving config to {config.config_path}", fg=typer.colors.GREEN)
    config.save()

    # create user records
    ms = MetadataStore(config)
    user_id = uuid.UUID(config.anon_clientid)
    user = User(
        id=uuid.UUID(config.anon_clientid),
        default_preset=default_preset,
        default_persona=default_persona,
        default_human=default_human,
        default_agent=default_agent,
        default_llm_config=LLMConfig(
            model=model,
            model_endpoint=model_endpoint,
            model_endpoint_type=model_endpoint_type,
            model_wrapper=model_wrapper,
            context_window=context_window,
        ),
        default_embedding_config=EmbeddingConfig(
            embedding_endpoint_type=embedding_endpoint_type,
            embedding_endpoint=embedding_endpoint,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            openai_key=openai_key,
            azure_key=azure_creds["azure_key"],
            azure_endpoint=azure_creds["azure_endpoint"],
            azure_version=azure_creds["azure_version"],
            azure_deployment=azure_creds["azure_deployment"],  # OK if None
        ),
    )
    if ms.get_user(user_id):
        # update user
        ms.update_user(user)
    else:
        ms.create_user(user)


class ListChoice(str, Enum):
    agents = "agents"
    humans = "humans"
    personas = "personas"
    sources = "sources"


@app.command()
def list(arg: Annotated[ListChoice, typer.Argument]):
    config = MemGPTConfig.load()
    ms = MetadataStore(config)
    user_id = uuid.UUID(config.anon_clientid)
    if arg == ListChoice.agents:
        """List all agents"""
        table = PrettyTable()
        table.field_names = ["Name", "Model", "Persona", "Human", "Data Source", "Create Time"]
        for agent in ms.list_agents(user_id=user_id):
            source_ids = ms.list_attached_sources(agent_id=agent.id)
            source_names = [ms.get_source(source_id=source_id).name for source_id in source_ids]
            table.add_row(
                [
                    agent.name,
                    agent.llm_config.model,
                    agent.persona,
                    agent.human,
                    ",".join(source_names),
                    utils.format_datetime(agent.created_at),
                ]
            )
        print(table)
    elif arg == ListChoice.humans:
        """List all humans"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for human_file in utils.list_human_files():
            text = open(human_file, "r").read()
            name = os.path.basename(human_file).replace("txt", "")
            table.add_row([name, text])
        print(table)
    elif arg == ListChoice.personas:
        """List all personas"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for persona_file in utils.list_persona_files():
            print(persona_file)
            text = open(persona_file, "r").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            table.add_row([name, text])
        print(table)
    elif arg == ListChoice.sources:
        """List all data sources"""

        # create table
        table = PrettyTable()
        table.field_names = ["Name", "Created At", "Agents"]
        # TODO: eventually look accross all storage connections
        # TODO: add data source stats
        # TODO: connect to agents

        # get all sources
        for source in ms.list_sources(user_id=user_id):
            # get attached agents
            agent_ids = ms.list_attached_agents(source_id=source.id)
            agent_names = [ms.get_agent(agent_id=agent_id).name for agent_id in agent_ids]

            table.add_row([source.name, utils.format_datetime(source.created_at), ",".join(agent_names)])

        print(table)
    else:
        raise ValueError(f"Unknown argument {arg}")


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
        raise ValueError(f"Unknown kind {option}")

    if filename:
        assert text is None, f"Cannot provide both filename and text"
        # copy file to directory
        shutil.copyfile(filename, os.path.join(directory, name))
    if text:
        assert filename is None, f"Cannot provide both filename and text"
        # write text to file
        with open(os.path.join(directory, name), "w") as f:
            f.write(text)


@app.command()
def delete(option: str, name: str):
    """Delete a source from the archival memory."""

    config = MemGPTConfig.load()
    user_id = uuid.UUID(config.anon_clientid)
    ms = MetadataStore(config)
    assert ms.get_user(user_id=user_id), f"User {user_id} does not exist"

    try:
        # delete from metadata
        if option == "source":
            # delete metadata
            source = ms.get_source(source_name=name, user_id=user_id)
            ms.delete_source(source_id=source.id)

            # delete from passages
            conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id=user_id)
            conn.delete({"data_source": name})

            assert (
                conn.get_all({"data_source": name}) == []
            ), f"Expected no passages with source {name}, but got {conn.get_all({'data_source': name})}"

            # TODO: should we also delete from agents?
        elif option == "agent":
            agent = ms.get_agent(agent_name=name, user_id=user_id)

            # recall memory
            recall_conn = StorageConnector.get_storage_connector(TableType.RECALL_MEMORY, config, user_id=user_id, agent_id=agent.id)
            recall_conn.delete({"agent_id": agent.id})

            # archival memory
            archival_conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config, user_id=user_id, agent_id=agent.id)
            archival_conn.delete({"agent_id": agent.id})

            # metadata
            ms.delete_agent(agent_id=agent.id)

        else:
            raise ValueError(f"Option {option} not implemented")

        typer.secho(f"Deleted source '{name}'", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Failed to deleted source '{name}'\n{e}", fg=typer.colors.RED)
