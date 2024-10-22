import ast
import os
from enum import Enum
from typing import Annotated, List, Optional

import questionary
import typer
from prettytable.colortable import ColorTable, Themes
from tqdm import tqdm

from letta import utils

app = typer.Typer()


@app.command()
def configure():
    """Updates default Letta configurations

    This function and quickstart should be the ONLY place where LettaConfig.save() is called
    """
    print("`letta configure` has been deprecated. Please see documentation on configuration, and run `letta run` instead.")


class ListChoice(str, Enum):
    agents = "agents"
    humans = "humans"
    personas = "personas"
    sources = "sources"


@app.command()
def list(arg: Annotated[ListChoice, typer.Argument]):
    from letta.client.client import create_client

    client = create_client()
    table = ColorTable(theme=Themes.OCEAN)
    if arg == ListChoice.agents:
        """List all agents"""
        table.field_names = ["Name", "LLM Model", "Embedding Model", "Embedding Dim", "Persona", "Human", "Data Source", "Create Time"]
        for agent in tqdm(client.list_agents()):
            # TODO: add this function
            sources = client.list_attached_sources(agent_id=agent.id)
            source_names = [source.name for source in sources if source is not None]
            table.add_row(
                [
                    agent.name,
                    agent.llm_config.model,
                    agent.embedding_config.embedding_model,
                    agent.embedding_config.embedding_dim,
                    agent.memory.get_block("persona").value[:100] + "...",
                    agent.memory.get_block("human").value[:100] + "...",
                    ",".join(source_names),
                    utils.format_datetime(agent.created_at),
                ]
            )
        print(table)
    elif arg == ListChoice.humans:
        """List all humans"""
        table.field_names = ["Name", "Text"]
        for human in client.list_humans():
            table.add_row([human.name, human.value.replace("\n", "")[:100]])
        print(table)
    elif arg == ListChoice.personas:
        """List all personas"""
        table.field_names = ["Name", "Text"]
        for persona in client.list_personas():
            table.add_row([persona.name, persona.value.replace("\n", "")[:100]])
        print(table)
    elif arg == ListChoice.sources:
        """List all data sources"""

        # create table
        table.field_names = ["Name", "Description", "Embedding Model", "Embedding Dim", "Created At"]
        # TODO: eventually look accross all storage connections
        # TODO: add data source stats
        # TODO: connect to agents

        # get all sources
        for source in client.list_sources():
            # get attached agents
            table.add_row(
                [
                    source.name,
                    source.description,
                    source.embedding_config.embedding_model,
                    source.embedding_config.embedding_dim,
                    utils.format_datetime(source.created_at),
                ]
            )

        print(table)
    else:
        raise ValueError(f"Unknown argument {arg}")
    return table


@app.command()
def add_tool(
    filename: str = typer.Option(..., help="Path to the Python file containing the function"),
    name: Optional[str] = typer.Option(None, help="Name of the tool"),
    update: bool = typer.Option(True, help="Update the tool if it already exists"),
    tags: Optional[List[str]] = typer.Option(None, help="Tags for the tool"),
):
    """Add or update a tool from a Python file."""
    from letta.client.client import create_client

    client = create_client()

    # 1. Parse the Python file
    with open(filename, "r", encoding="utf-8") as file:
        source_code = file.read()

    # 2. Parse the source code to extract the function
    # Note: here we assume it is one function only in the file.
    module = ast.parse(source_code)
    func_def = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if not func_def:
        raise ValueError("No function found in the provided file")

    # 3. Compile the function to make it callable
    # Explanation courtesy of GPT-4:
    # Compile the AST (Abstract Syntax Tree) node representing the function definition into a code object
    # ast.Module creates a module node containing the function definition (func_def)
    # compile converts the AST into a code object that can be executed by the Python interpreter
    # The exec function executes the compiled code object in the current context,
    # effectively defining the function within the current namespace
    exec(compile(ast.Module([func_def], []), filename, "exec"))
    # Retrieve the function object by evaluating its name in the current namespace
    # eval looks up the function name in the current scope and returns the function object
    func = eval(func_def.name)

    # 4. Add or update the tool
    tool = client.create_tool(func=func, name=name, tags=tags, update=update)
    print(f"Tool {tool.name} added successfully")


@app.command()
def list_tools():
    """List all available tools."""
    from letta.client.client import create_client

    client = create_client()

    tools = client.list_tools()
    for tool in tools:
        print(f"Tool: {tool.name}")


@app.command()
def add(
    option: str,  # [human, persona]
    name: Annotated[str, typer.Option(help="Name of human/persona")],
    text: Annotated[Optional[str], typer.Option(help="Text of human/persona")] = None,
    filename: Annotated[Optional[str], typer.Option("-f", help="Specify filename")] = None,
):
    """Add a person/human"""
    from letta.client.client import create_client

    client = create_client(base_url=os.getenv("MEMGPT_BASE_URL"), token=os.getenv("MEMGPT_SERVER_PASS"))
    if filename:  # read from file
        assert text is None, "Cannot specify both text and filename"
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        assert text is not None, "Must specify either text or filename"
    if option == "persona":
        persona_id = client.get_persona_id(name)
        if persona_id:
            client.get_persona(persona_id)
            # config if user wants to overwrite
            if not questionary.confirm(f"Persona {name} already exists. Overwrite?").ask():
                return
            client.update_persona(persona_id, text=text)
        else:
            client.create_persona(name=name, text=text)

    elif option == "human":
        human_id = client.get_human_id(name)
        if human_id:
            human = client.get_human(human_id)
            # config if user wants to overwrite
            if not questionary.confirm(f"Human {name} already exists. Overwrite?").ask():
                return
            client.update_human(human_id, text=text)
        else:
            human = client.create_human(name=name, text=text)
    else:
        raise ValueError(f"Unknown kind {option}")


@app.command()
def delete(option: str, name: str):
    """Delete a source from the archival memory."""
    from letta.client.client import create_client

    client = create_client(base_url=os.getenv("MEMGPT_BASE_URL"), token=os.getenv("MEMGPT_API_KEY"))
    try:
        # delete from metadata
        if option == "source":
            # delete metadata
            source_id = client.get_source_id(name)
            assert source_id is not None, f"Source {name} does not exist"
            client.delete_source(source_id)
        elif option == "agent":
            agent_id = client.get_agent_id(name)
            assert agent_id is not None, f"Agent {name} does not exist"
            client.delete_agent(agent_id=agent_id)
        elif option == "human":
            human_id = client.get_human_id(name)
            assert human_id is not None, f"Human {name} does not exist"
            client.delete_human(human_id)
        elif option == "persona":
            persona_id = client.get_persona_id(name)
            assert persona_id is not None, f"Persona {name} does not exist"
            client.delete_persona(persona_id)
        else:
            raise ValueError(f"Option {option} not implemented")

        typer.secho(f"Deleted {option} '{name}'", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Failed to delete {option}'{name}'\n{e}", fg=typer.colors.RED)
