"""
This file contains functions for loading data into MemGPT's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
memgpt load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

from typing import List, Optional, Annotated
from tqdm import tqdm
import numpy as np
import typer
import uuid

from memgpt.data_sources.connectors import load_data, DirectoryConnector, VectorDBConnector
from memgpt.embeddings import embedding_model, check_and_split_text
from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.metadata import MetadataStore
from memgpt.data_types import Source, Passage, Document, User
from memgpt.utils import get_utc_time, suppress_stdout
from memgpt.agent_store.storage import StorageConnector, TableType


app = typer.Typer()

# NOTE: not supported due to llama-index breaking things (please reach out if you still need it)
# @app.command("index")
# def load_index(
#    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
#    dir: Annotated[Optional[str], typer.Option(help="Path to directory containing index.")] = None,
#    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
# ):
#    """Load a LlamaIndex saved VectorIndex into MemGPT"""
#    if user_id is None:
#        config = MemGPTConfig.load()
#        user_id = uuid.UUID(config.anon_clientid)
#
#    try:
#        # load index data
#        storage_context = StorageContext.from_defaults(persist_dir=dir)
#        loaded_index = load_index_from_storage(storage_context)
#
#        # hacky code to extract out passages/embeddings (thanks a lot, llama index)
#        embed_dict = loaded_index._vector_store._data.embedding_dict
#        node_dict = loaded_index._docstore.docs
#
#        # create storage connector
#        config = MemGPTConfig.load()
#        if user_id is None:
#            user_id = uuid.UUID(config.anon_clientid)
#
#        passages = []
#        for node_id, node in node_dict.items():
#            vector = embed_dict[node_id]
#            node.embedding = vector
#            # assume embedding are the same as config
#            passages.append(
#                Passage(
#                    text=node.text,
#                    embedding=np.array(vector),
#                    embedding_dim=config.default_embedding_config.embedding_dim,
#                    embedding_model=config.default_embedding_config.embedding_model,
#                )
#            )
#            assert config.default_embedding_config.embedding_dim == len(
#                vector
#            ), f"Expected embedding dimension {config.default_embedding_config.embedding_dim}, got {len(vector)}"
#
#        if len(passages) == 0:
#            raise ValueError(f"No passages found in index {dir}")
#
#        insert_passages_into_source(passages, name, user_id, config)
#    except ValueError as e:
#        typer.secho(f"Failed to load index from provided information.\n{e}", fg=typer.colors.RED)


default_extensions = ".txt,.md,.pdf"


@app.command("directory")
def load_directory(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    input_dir: Annotated[Optional[str], typer.Option(help="Path to directory containing dataset.")] = None,
    input_files: Annotated[List[str], typer.Option(help="List of paths to files containing dataset.")] = [],
    recursive: Annotated[bool, typer.Option(help="Recursively search for files in directory.")] = False,
    extensions: Annotated[str, typer.Option(help="Comma separated list of file extensions to load")] = default_extensions,
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,  # TODO: remove
):
    try:
        connector = DirectoryConnector(input_files=input_files, input_directory=input_dir, recursive=recursive, extensions=extensions)
        config = MemGPTConfig.load()
        if not user_id:
            user_id = uuid.UUID(config.anon_clientid)

        ms = MetadataStore(config)
        source = Source(
            name=name,
            user_id=user_id,
            embedding_model=config.default_embedding_config.embedding_model,
            embedding_dim=config.default_embedding_config.embedding_dim,
        )
        ms.create_source(source)
        passage_storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
        # TODO: also get document store

        # ingest data into passage/document store
        try:
            num_passages, num_documents = load_data(
                connector=connector,
                source=source,
                embedding_config=config.default_embedding_config,
                document_store=None,
                passage_store=passage_storage,
            )
            print(f"Loaded {num_passages} passages and {num_documents} documents from {name}")
        except Exception as e:
            typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
            ms.delete_source(source_id=source.id)

    except ValueError as e:
        typer.secho(f"Failed to load directory from provided information.\n{e}", fg=typer.colors.RED)
        raise


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
#
#
# @app.command("database")
# def load_database(
#    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
#    query: Annotated[str, typer.Option(help="Database query.")],
#    dump_path: Annotated[Optional[str], typer.Option(help="Path to dump file.")] = None,
#    scheme: Annotated[Optional[str], typer.Option(help="Database scheme.")] = None,
#    host: Annotated[Optional[str], typer.Option(help="Database host.")] = None,
#    port: Annotated[Optional[int], typer.Option(help="Database port.")] = None,
#    user: Annotated[Optional[str], typer.Option(help="Database user.")] = None,
#    password: Annotated[Optional[str], typer.Option(help="Database password.")] = None,
#    dbname: Annotated[Optional[str], typer.Option(help="Database name.")] = None,
# ):
#    try:
#        from llama_index.readers.database import DatabaseReader
#
#        print(dump_path, scheme)
#
#        if dump_path is not None:
#            # read from database dump file
#            from sqlalchemy import create_engine
#
#            engine = create_engine(f"sqlite:///{dump_path}")
#
#            db = DatabaseReader(engine=engine)
#        else:
#            assert dump_path is None, "Cannot provide both dump_path and database connection parameters."
#            assert scheme is not None, "Must provide database scheme."
#            assert host is not None, "Must provide database host."
#            assert port is not None, "Must provide database port."
#            assert user is not None, "Must provide database user."
#            assert password is not None, "Must provide database password."
#            assert dbname is not None, "Must provide database name."
#
#            db = DatabaseReader(
#                scheme=scheme,  # Database Scheme
#                host=host,  # Database Host
#                port=str(port),  # Database Port
#                user=user,  # Database User
#                password=password,  # Database Password
#                dbname=dbname,  # Database Name
#            )
#
#        # load data
#        docs = db.load_data(query=query)
#        store_docs(name, docs)
#    except ValueError as e:
#        typer.secho(f"Failed to load database from provided information.\n{e}", fg=typer.colors.RED)
#


@app.command("vector-database")
def load_vector_database(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    uri: Annotated[str, typer.Option(help="Database URI.")],
    table_name: Annotated[str, typer.Option(help="Name of table containing data.")],
    text_column: Annotated[str, typer.Option(help="Name of column containing text.")],
    embedding_column: Annotated[str, typer.Option(help="Name of column containing embedding.")],
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
):
    """Load pre-computed embeddings into MemGPT from a database."""
    try:
        config = MemGPTConfig.load()
        connector = VectorDBConnector(
            uri=uri,
            table_name=table_name,
            text_column=text_column,
            embedding_column=embedding_column,
            embedding_dim=config.default_embedding_config.embedding_dim,
        )
        if not user_id:
            user_id = uuid.UUID(config.anon_clientid)

        ms = MetadataStore(config)
        source = Source(
            name=name,
            user_id=user_id,
            embedding_model=config.default_embedding_config.embedding_model,
            embedding_dim=config.default_embedding_config.embedding_dim,
        )
        ms.create_source(source)
        passage_storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
        # TODO: also get document store

        # ingest data into passage/document store
        try:
            num_passages, num_documents = load_data(
                connector=connector,
                source=source,
                embedding_config=config.default_embedding_config,
                document_store=None,
                passage_store=passage_storage,
            )
            print(f"Loaded {num_passages} passages and {num_documents} documents from {name}")
        except Exception as e:
            typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
            ms.delete_source(source_id=source.id)

    except ValueError as e:
        typer.secho(f"Failed to load VectorDB from provided information.\n{e}", fg=typer.colors.RED)
        raise
