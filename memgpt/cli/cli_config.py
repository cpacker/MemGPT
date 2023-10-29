import questionary
import openai
from prettytable import PrettyTable
import typer
import os
import shutil

# from memgpt.cli import app
from memgpt import utils
from memgpt.config import MemGPTConfig, AgentConfig

app = typer.Typer()


@app.command()
def configure():
    """Updates default MemGPT configurations"""

    default_provider = "openai"

    # openai credentials
    use_openai = questionary.confirm("Do you want to enable MemGPT with Open AI?").ask()
    if use_openai:
        # search for key in enviornment
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            openai_key = questionary.text("Open AI keys not found in enviornment - please enter:").ask()
        default_openai = questionary.confirm("Use OpenAI as default provider?").ask()
        if default_openai:
            default_provider = "openai"

    # azure credentials
    use_azure = questionary.confirm("Do you want to enable MemGPT with Azure?").ask()
    use_azure_deployment_ids = False
    if use_azure:
        # search for key in enviornment
        azure_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = (os.getenv("AZURE_ENDPOINT"),)
        azure_version = (os.getenv("AZURE_VERSION"),)
        azure_deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT"),)
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        if all([azure_key, azure_endpoint, azure_version]):
            print(f"Using Microsoft endpoint {azure_endpoint}.")
            if all([azure_deployment, azure_embedding_deployment]):
                print(f"Using deployment id {azure_deployment}")
                use_azure_deployment_ids = True
            default_azure = questionary.confirm("Use Azure as default provider?").ask()
            if default_azure:
                default_provider = "azure"

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

    # default model
    model_options = []
    if use_openai:
        model_options += ["gpt-3.5-turbo", "gpt-3.5", "gpt-4"]
    default_model = questionary.select(
        "Select default model (recommended: gpt-4):", choices=["gpt-3.5-turbo", "gpt-3.5", "gpt-4"], default="gpt-4"
    ).ask()

    # defaults
    personas = [os.path.basename(f).replace(".txt", "") for f in utils.list_persona_files()]
    print(personas)
    default_persona = questionary.select("Select default persona:", personas, default="sam_pov").ask()
    humans = [os.path.basename(f).replace(".txt", "") for f in utils.list_human_files()]
    print(humans)
    default_human = questionary.select("Select default human:", humans, default="cs_phd").ask()

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

    # TODO: allow configuring embedding model

    config = MemGPTConfig(
        model=default_model,
        provider=default_provider,
        default_persona=default_persona,
        default_human=default_human,
        default_agent=default_agent,
        openai_key=openai_key if use_openai else None,
        azure_key=azure_key if use_azure else None,
        azure_endpoint=azure_endpoint if use_azure else None,
        azure_version=azure_version if use_azure else None,
        azure_deployment=azure_deployment if use_azure_deployment_ids else None,
        azure_embedding_deployment=azure_embedding_deployment if use_azure_deployment_ids else None,
    )
    print(f"Saving config to {config.config_path}")
    config.save()


@app.command()
def list(option: str):

    print("new list")

    if option == "agents":
        """List all agents"""
        table = PrettyTable()
        table.field_names = ["Name", "Model", "Persona", "Human", "Data Source"]
        for agent_file in utils.list_agent_config_files():
            agent_name = os.path.basename(agent_file).replace(".json", "")
            agent_config = AgentConfig.load(agent_name)
            table.add_row([agent_name, agent_config.model, agent_config.persona, agent_config.human, agent_config.data_source])
        print(table)
    elif option == "humans":
        """List all humans"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for human_file in utils.list_human_files():
            name = os.path.basename(human_file)
            text = humans.get_human_text(name)
            table.add_row([name, text])
        print(table)
    elif option == "personas":
        """List all personas"""
        table.field_names = ["Name", "Text"]
        for persona_file in utils.list_persona_files():
            name = os.path.basename(persona_file)
            text = personas.get_persona_text(name)
            table.add_row([name, text])
        print(table)
    elif option == "sources":
        """List all data sources"""
        table = PrettyTable()
        table.field_names = ["Name", "Create Time", "Agents"]
        for data_source_file in os.listdir(os.path.join(MEMGPT_DIR, "archival")):
            name = os.path.basename(data_source_file)
            table.add_row([name, "TODO", "TODO"])
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
