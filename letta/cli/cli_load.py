"""
This file contains functions for loading data into Letta's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
letta load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

import uuid
from typing import Annotated, List, Optional

import questionary
import typer

from letta import create_client
from letta.data_sources.connectors import DirectoryConnector

app = typer.Typer()


default_extensions = ".txt,.md,.pdf"


@app.command("directory")
def load_directory(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    input_dir: Annotated[Optional[str], typer.Option(help="Path to directory containing dataset.")] = None,
    input_files: Annotated[List[str], typer.Option(help="List of paths to files containing dataset.")] = [],
    recursive: Annotated[bool, typer.Option(help="Recursively search for files in directory.")] = False,
    extensions: Annotated[str, typer.Option(help="Comma separated list of file extensions to load")] = default_extensions,
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,  # TODO: remove
    description: Annotated[Optional[str], typer.Option(help="Description of the source.")] = None,
):
    client = create_client()

    # create connector
    connector = DirectoryConnector(input_files=input_files, input_directory=input_dir, recursive=recursive, extensions=extensions)

    # choose form list of embedding configs
    embedding_configs = client.list_embedding_configs()
    embedding_options = [embedding_config.embedding_model for embedding_config in embedding_configs]

    embedding_choices = [
        questionary.Choice(title=embedding_config.pretty_print(), value=embedding_config) for embedding_config in embedding_configs
    ]

    # select model
    if len(embedding_options) == 0:
        raise ValueError("No embedding models found. Please enable a provider.")
    elif len(embedding_options) == 1:
        embedding_model_name = embedding_options[0]
    else:
        embedding_model_name = questionary.select("Select embedding model:", choices=embedding_choices).ask().embedding_model
    embedding_config = [
        embedding_config for embedding_config in embedding_configs if embedding_config.embedding_model == embedding_model_name
    ][0]

    # create source
    source = client.create_source(name=name, embedding_config=embedding_config)

    # load data
    try:
        client.load_data(connector, source_name=name)
    except Exception as e:
        typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
        client.delete_source(source.id)
