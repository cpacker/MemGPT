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


@app.command()
def configure():
    """Updates default MemGPT configurations"""

    from memgpt.presets import DEFAULT_PRESET, preset_options

    MemGPTConfig.create_config_dir()

    # Will pre-populate with defaults, or what the user previously set
    config = MemGPTConfig.load()

    # openai credentials
    use_openai = questionary.confirm("Do you want to enable MemGPT with OpenAI?", default=True).ask()
    if use_openai:
        # search for key in enviornment
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("Missing enviornment variables for OpenAI. Please set them and run `memgpt configure` again.")
            # TODO: eventually stop relying on env variables and pass in keys explicitly
            # openai_key = questionary.text("Open AI API keys not found in enviornment - please enter:").ask()

    # azure credentials
    use_azure = questionary.confirm("Do you want to enable MemGPT with Azure?", default=(config.azure_key is not None)).ask()
    use_azure_deployment_ids = False
    if use_azure:
        # search for key in enviornment
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_version = os.getenv("AZURE_OPENAI_VERSION")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

        if all([azure_key, azure_endpoint, azure_version]):
            print(f"Using Microsoft endpoint {azure_endpoint}.")
            if all([azure_deployment, azure_embedding_deployment]):
                print(f"Using deployment id {azure_deployment}")
                use_azure_deployment_ids = True

            # configure openai
            openai.api_type = "azure"
            openai.api_key = azure_key
            openai.api_base = azure_endpoint
            openai.api_version = azure_version
        else:
            print("Missing enviornment variables for Azure. Please set then run `memgpt configure` again.")
            # TODO: allow for manual setting
            use_azure = False

    # TODO: configure local model

    # configure provider
    model_endpoint_options = []
    if os.getenv("OPENAI_API_BASE") is not None:
        model_endpoint_options.append(os.getenv("OPENAI_API_BASE"))
    if use_openai:
        model_endpoint_options += ["openai"]
    if use_azure:
        model_endpoint_options += ["azure"]
    assert (
        len(model_endpoint_options) > 0
    ), "No endpoints found. Please enable OpenAI, Azure, or set OPENAI_API_BASE to point at the IP address of your LLM server."
    valid_default_model = config.model_endpoint in model_endpoint_options
    default_endpoint = questionary.select(
        "Select default inference endpoint:",
        model_endpoint_options,
        default=config.model_endpoint if valid_default_model else model_endpoint_options[0],
    ).ask()

    # configure embedding provider
    embedding_endpoint_options = []
    if use_azure:
        embedding_endpoint_options += ["azure"]
    if use_openai:
        embedding_endpoint_options += ["openai"]
    embedding_endpoint_options += ["local"]
    valid_default_embedding = config.embedding_model in embedding_endpoint_options
    # determine the default selection in a smart way
    if "openai" in embedding_endpoint_options and default_endpoint == "openai":
        # openai llm -> openai embeddings
        default_embedding_endpoint_default = "openai"
    elif default_endpoint not in ["openai", "azure"]:  # is local
        # local llm -> local embeddings
        default_embedding_endpoint_default = "local"
    else:
        default_embedding_endpoint_default = config.embedding_model if valid_default_embedding else embedding_endpoint_options[-1]
    default_embedding_endpoint = questionary.select(
        "Select default embedding endpoint:", embedding_endpoint_options, default=default_embedding_endpoint_default
    ).ask()

    # configure embedding dimentions
    default_embedding_dim = config.embedding_dim
    if default_embedding_endpoint == "local":
        # HF model uses lower dimentionality
        default_embedding_dim = 384

    # configure preset
    default_preset = questionary.select("Select default preset:", preset_options, default=config.preset).ask()

    # default model
    if use_openai or use_azure:
        model_options = []
        if use_openai:
            model_options += ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo-16k"]
        default_model = questionary.select(
            "Select default model (recommended: gpt-4):", choices=["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo-16k"], default=config.model
        ).ask()
    else:
        default_model = "local"  # TODO: figure out if this is ok? this is for local endpoint

    # get the max tokens (context window) for the model
    if default_model == "local" or str(default_model) not in LLM_MAX_TOKENS:
        # Ask the user to specify the context length
        context_length_options = [
            str(2**12),  # 4096
            str(2**13),  # 8192
            str(2**14),  # 16384
            str(2**15),  # 32768
            str(2**18),  # 262144
            "custom",  # enter yourself
        ]
        default_model_context_window = questionary.select(
            "Select your model's context window (for Mistral 7B models, this is probably 8k / 8192):",
            choices=context_length_options,
            default=str(LLM_MAX_TOKENS["DEFAULT"]),
        ).ask()

        # If custom, ask for input
        if default_model_context_window == "custom":
            while True:
                default_model_context_window = questionary.text("Enter context window (e.g. 8192)").ask()
                try:
                    default_model_context_window = int(default_model_context_window)
                    break
                except ValueError:
                    print(f"Context window must be a valid integer")
        else:
            default_model_context_window = int(default_model_context_window)
    else:
        # Pull the context length from the models
        default_model_context_window = LLM_MAX_TOKENS[default_model]

    # defaults
    personas = [os.path.basename(f).replace(".txt", "") for f in utils.list_persona_files()]
    # print(personas)
    default_persona = questionary.select("Select default persona:", personas, default=config.default_persona).ask()
    humans = [os.path.basename(f).replace(".txt", "") for f in utils.list_human_files()]
    # print(humans)
    default_human = questionary.select("Select default human:", humans, default=config.default_human).ask()

    # TODO: figure out if we should set a default agent or not
    default_agent = None
    # agents = [os.path.basename(f).replace(".json", "") for f in utils.list_agent_config_files()]
    # if len(agents) > 0: # agents have been created
    #    default_agent = questionary.select(
    #        "Select default agent:",
    #        agents
    #    ).ask()
    # else:
    #    default_agent = None

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

    # TODO: allow configuring embedding model

    config = MemGPTConfig(
        model=default_model,
        context_window=default_model_context_window,
        preset=default_preset,
        model_endpoint=default_endpoint,
        embedding_model=default_embedding_endpoint,
        embedding_dim=default_embedding_dim,
        default_persona=default_persona,
        default_human=default_human,
        default_agent=default_agent,
        openai_key=openai_key if use_openai else None,
        azure_key=azure_key if use_azure else None,
        azure_endpoint=azure_endpoint if use_azure else None,
        azure_version=azure_version if use_azure else None,
        azure_deployment=azure_deployment if use_azure_deployment_ids else None,
        azure_embedding_deployment=azure_embedding_deployment if use_azure_deployment_ids else None,
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
