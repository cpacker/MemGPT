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

from memgpt.embeddings import embedding_model, check_and_split_text
from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.metadata import MetadataStore
from memgpt.data_types import Source, Passage, Document, User
from memgpt.utils import get_utc_time, suppress_stdout
from memgpt.agent_store.storage import StorageConnector, TableType

from datetime import datetime

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

app = typer.Typer()


def insert_passages_into_source(passages: List[Passage], source_name: str, user_id: uuid.UUID, config: MemGPTConfig):
    """Insert a list of passages into a source by updating storage connectors and metadata store"""
    storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    orig_size = storage.size()

    # insert metadata store
    ms = MetadataStore(config)
    source = ms.get_source(user_id=user_id, source_name=source_name)
    if not source:
        # create new
        source = Source(user_id=user_id, name=source_name)
        ms.create_source(source)

    # make sure user_id is set for passages
    for passage in passages:
        # TODO: attach source IDs
        # passage.source_id = source.id
        passage.user_id = user_id
        passage.data_source = source_name

    # add and save all passages
    storage.insert_many(passages)

    assert orig_size + len(passages) == storage.size(), f"Expected {orig_size + len(passages)} passages, got {storage.size()}"
    storage.save()


def store_docs(name, docs, user_id=None, show_progress=True):
    """Common function for embedding and storing documents"""

    config = MemGPTConfig.load()
    if user_id is None:  # assume running local with single user
        user_id = uuid.UUID(config.anon_clientid)

    # ensure doc text is not too long
    # TODO: replace this to instead split up docs that are too large
    # (this is a temporary fix to avoid breaking the llama index)
    for doc in docs:
        doc.text = check_and_split_text(doc.text, config.default_embedding_config.embedding_model)[0]

    # record data source metadata
    ms = MetadataStore(config)
    user = ms.get_user(user_id)
    if user is None:
        raise ValueError(f"Cannot find user {user_id} in metadata store. Please run 'memgpt configure'.")
    # create data source record
    data_source = Source(
        user_id=user.id,
        name=name,
        embedding_model=config.default_embedding_config.embedding_model,
        embedding_dim=config.default_embedding_config.embedding_dim,
    )
    existing_source = ms.get_source(user_id=user.id, source_name=name)
    if not existing_source:
        ms.create_source(data_source)
    else:
        print(f"Source {name} for user {user.id} already exists.")
        if existing_source.embedding_model != data_source.embedding_model:
            print(
                f"Warning: embedding model for existing source {existing_source.embedding_model} does not match default {data_source.embedding_model}"
            )
            print("Cannot import data into this source without a compatible embedding endpoint.")
            print("Please run 'memgpt configure' to update the default embedding settings.")
            return False
        if existing_source.embedding_dim != data_source.embedding_dim:
            print(
                f"Warning: embedding dimension for existing source {existing_source.embedding_dim} does not match default {data_source.embedding_dim}"
            )
            print("Cannot import data into this source without a compatible embedding endpoint.")
            print("Please run 'memgpt configure' to update the default embedding settings.")
            return False

    # compute and record passages
    embed_model = embedding_model(config.default_embedding_config)

    # use llama index to run embeddings code
    with suppress_stdout():
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=embed_model, chunk_size=config.default_embedding_config.embedding_chunk_size
        )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)
    embed_dict = index._vector_store._data.embedding_dict
    node_dict = index._docstore.docs

    # TODO: add document store

    # gather passages
    passages = []
    for node_id, node in tqdm(node_dict.items()):
        vector = embed_dict[node_id]
        node.embedding = vector
        text = node.text.replace("\x00", "\uFFFD")  # hacky fix for error on null characters
        assert (
            len(node.embedding) == config.default_embedding_config.embedding_dim
        ), f"Expected embedding dimension {config.default_embedding_config.embedding_dim}, got {len(node.embedding)}: {node.embedding}"
        passages.append(
            Passage(
                user_id=user.id,
                text=text,
                data_source=name,
                embedding=node.embedding,
                metadata=None,
                embedding_dim=config.default_embedding_config.embedding_dim,
                embedding_model=config.default_embedding_config.embedding_model,
            )
        )

    insert_passages_into_source(passages, name, user_id, config)


@app.command("index")
def load_index(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    dir: Annotated[Optional[str], typer.Option(help="Path to directory containing index.")] = None,
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
):
    """Load a LlamaIndex saved VectorIndex into MemGPT"""
    if user_id is None:
        config = MemGPTConfig.load()
        user_id = uuid.UUID(config.anon_clientid)

    try:
        # load index data
        storage_context = StorageContext.from_defaults(persist_dir=dir)
        loaded_index = load_index_from_storage(storage_context)

        # hacky code to extract out passages/embeddings (thanks a lot, llama index)
        embed_dict = loaded_index._vector_store._data.embedding_dict
        node_dict = loaded_index._docstore.docs

        # create storage connector
        config = MemGPTConfig.load()
        if user_id is None:
            user_id = uuid.UUID(config.anon_clientid)

        passages = []
        for node_id, node in node_dict.items():
            vector = embed_dict[node_id]
            node.embedding = vector
            # assume embedding are the same as config
            passages.append(
                Passage(
                    text=node.text,
                    embedding=np.array(vector),
                    embedding_dim=config.default_embedding_config.embedding_dim,
                    embedding_model=config.default_embedding_config.embedding_model,
                )
            )
            assert config.default_embedding_config.embedding_dim == len(
                vector
            ), f"Expected embedding dimension {config.default_embedding_config.embedding_dim}, got {len(vector)}"

        if len(passages) == 0:
            raise ValueError(f"No passages found in index {dir}")

        insert_passages_into_source(passages, name, user_id, config)
    except ValueError as e:
        typer.secho(f"Failed to load index from provided information.\n{e}", fg=typer.colors.RED)


default_extensions = ".txt,.md,.pdf"


@app.command("directory")
def load_directory(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    input_dir: Annotated[Optional[str], typer.Option(help="Path to directory containing dataset.")] = None,
    input_files: Annotated[List[str], typer.Option(help="List of paths to files containing dataset.")] = [],
    recursive: Annotated[bool, typer.Option(help="Recursively search for files in directory.")] = False,
    extensions: Annotated[str, typer.Option(help="Comma separated list of file extensions to load")] = default_extensions,
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
):
    try:
        from llama_index import SimpleDirectoryReader

        if recursive == True:
            assert input_dir is not None, "Must provide input directory if recursive is True."

        if input_dir is not None:
            reader = SimpleDirectoryReader(
                input_dir=str(input_dir),
                recursive=recursive,
                required_exts=[ext.strip() for ext in str(extensions).split(",")],
            )
        else:
            assert input_files is not None, "Must provide input files if input_dir is None"
            reader = SimpleDirectoryReader(input_files=[str(f) for f in input_files])

        # load docs
        docs = reader.load_data()
        store_docs(str(name), docs, user_id)

    except ValueError as e:
        typer.secho(f"Failed to load directory from provided information.\n{e}", fg=typer.colors.RED)
        raise


@app.command("webpage")
def load_webpage(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    urls: Annotated[List[str], typer.Option(help="List of urls to load.")],
):
    try:
        from llama_index.readers.web import SimpleWebPageReader

        docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
        store_docs(name, docs)

    except ValueError as e:
        typer.secho(f"Failed to load webpage from provided information.\n{e}", fg=typer.colors.RED)


@app.command("database")
def load_database(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    query: Annotated[str, typer.Option(help="Database query.")],
    dump_path: Annotated[Optional[str], typer.Option(help="Path to dump file.")] = None,
    scheme: Annotated[Optional[str], typer.Option(help="Database scheme.")] = None,
    host: Annotated[Optional[str], typer.Option(help="Database host.")] = None,
    port: Annotated[Optional[int], typer.Option(help="Database port.")] = None,
    user: Annotated[Optional[str], typer.Option(help="Database user.")] = None,
    password: Annotated[Optional[str], typer.Option(help="Database password.")] = None,
    dbname: Annotated[Optional[str], typer.Option(help="Database name.")] = None,
):
    try:
        from llama_index.readers.database import DatabaseReader

        print(dump_path, scheme)

        if dump_path is not None:
            # read from database dump file
            from sqlalchemy import create_engine

            engine = create_engine(f"sqlite:///{dump_path}")

            db = DatabaseReader(engine=engine)
        else:
            assert dump_path is None, "Cannot provide both dump_path and database connection parameters."
            assert scheme is not None, "Must provide database scheme."
            assert host is not None, "Must provide database host."
            assert port is not None, "Must provide database port."
            assert user is not None, "Must provide database user."
            assert password is not None, "Must provide database password."
            assert dbname is not None, "Must provide database name."

            db = DatabaseReader(
                scheme=scheme,  # Database Scheme
                host=host,  # Database Host
                port=str(port),  # Database Port
                user=user,  # Database User
                password=password,  # Database Password
                dbname=dbname,  # Database Name
            )

        # load data
        docs = db.load_data(query=query)
        store_docs(name, docs)
    except ValueError as e:
        typer.secho(f"Failed to load database from provided information.\n{e}", fg=typer.colors.RED)


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
    if user_id is None:
        config = MemGPTConfig.load()
        user_id = uuid.UUID(config.anon_clientid)

    try:
        from sqlalchemy import create_engine, select, MetaData, Table, Inspector
        from pgvector.sqlalchemy import Vector

        # connect to db table
        engine = create_engine(uri)
        metadata = MetaData()
        # Create an inspector to inspect the database
        inspector = Inspector.from_engine(engine)
        table_names = inspector.get_table_names()
        assert table_name in table_names, f"Table {table_name} not found in database: tables that exist {table_names}."

        table = Table(table_name, metadata, autoload_with=engine)

        config = MemGPTConfig.load()

        # Prepare a select statement
        select_statement = select(
            table.c[text_column], table.c[embedding_column].cast(Vector(config.default_embedding_config.embedding_dim))
        )

        # Execute the query and fetch the results
        with engine.connect() as connection:
            result = connection.execute(select_statement).fetchall()

        # Convert to a list of tuples (text, embedding)
        passages = []
        for text, embedding in result:
            # assume that embeddings are the same model as in config
            passages.append(
                Passage(
                    text=text,
                    embedding=embedding,
                    user_id=user_id,
                    embedding_dim=config.default_embedding_config.embedding_dim,
                    embedding_model=config.default_embedding_config.embedding_model,
                )
            )
            assert config.default_embedding_config.embedding_dim == len(
                embedding
            ), f"Expected embedding dimension {config.default_embedding_config.embedding_dim}, got {len(embedding)}"

        # create storage connector
        config = MemGPTConfig.load()
        if user_id is None:
            user_id = uuid.UUID(config.anon_clientid)

        insert_passages_into_source(passages, name, user_id, config)

    except ValueError as e:
        typer.secho(f"Failed to load vector database from provided information.\n{e}", fg=typer.colors.RED)
