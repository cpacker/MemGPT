"""
This file contains functions for loading data into Letta's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
letta load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

import uuid
from typing import Annotated, List, Optional

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

    # create source
    source = client.create_source(name=name)

    # load data
    try:
        client.load_data(connector, source_name=name)
    except Exception as e:
        typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
        client.delete_source(source.id)


# @app.command("webpage")
# def load_webpage(
#    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
#    urls: Annotated[List[str], typer.Option(help="List of urls to load.")],
# ):
#    try:
#        from llama_index.readers.web import SimpleWebPageReader
#
#        docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
#        store_docs(name, docs)
#
#    except ValueError as e:
#        typer.secho(f"Failed to load webpage from provided information.\n{e}", fg=typer.colors.RED)


@app.command("vector-database")
def load_vector_database(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    uri: Annotated[str, typer.Option(help="Database URI.")],
    table_name: Annotated[str, typer.Option(help="Name of table containing data.")],
    text_column: Annotated[str, typer.Option(help="Name of column containing text.")],
    embedding_column: Annotated[str, typer.Option(help="Name of column containing embedding.")],
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
):
    """Load pre-computed embeddings into Letta from a database."""
    raise NotImplementedError
    # try:
    #    config = LettaConfig.load()
    #    connector = VectorDBConnector(
    #        uri=uri,
    #        table_name=table_name,
    #        text_column=text_column,
    #        embedding_column=embedding_column,
    #        embedding_dim=config.default_embedding_config.embedding_dim,
    #    )
    #    if not user_id:
    #        user_id = uuid.UUID(config.anon_clientid)

    #    ms = MetadataStore(config)
    #    source = Source(
    #        name=name,
    #        user_id=user_id,
    #        embedding_model=config.default_embedding_config.embedding_model,
    #        embedding_dim=config.default_embedding_config.embedding_dim,
    #    )
    #    ms.create_source(source)
    #    passage_storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    #    # TODO: also get document store

    #    # ingest data into passage/document store
    #    try:
    #        num_passages, num_documents = load_data(
    #            connector=connector,
    #            source=source,
    #            embedding_config=config.default_embedding_config,
    #            document_store=None,
    #            passage_store=passage_storage,
    #        )
    #        print(f"Loaded {num_passages} passages and {num_documents} files from {name}")
    #    except Exception as e:
    #        typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
    #        ms.delete_source(source_id=source.id)

    # except ValueError as e:
    #    typer.secho(f"Failed to load VectorDB from provided information.\n{e}", fg=typer.colors.RED)
    #    raise
