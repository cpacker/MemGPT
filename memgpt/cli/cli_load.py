"""
This file contains functions for loading data into MemGPT's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
memgpt load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

from typing import List

import llama_index
from llama_index.vector_stores import VectorStoreQuery, SimpleVectorStore
from tqdm import tqdm
import numpy as np
import typer
import uuid
from memgpt.embeddings import embedding_model, check_and_split_text
from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.metadata import MetadataStore
from memgpt.data_types import Source, Passage, Document, User
from memgpt.utils import get_local_time, suppress_stdout
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
        source = Source(user_id=user_id, name=source_name, created_at=get_local_time())
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

    # record data source metadata
    ms = MetadataStore(config)
    user = ms.get_user(user_id)
    if user is None:
        raise ValueError(f"Cannot find user {user_id} in metadata store. Please run 'memgpt configure'.")

    # create data source record
    data_source = Source(
        user_id=user.id,
        name=name,
        created_at=datetime.now(),
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

    storage = StorageConnector.get_storage_connector(TableType.DOCUMENTS, config, user_id)
    docs_storage = []
    for doc in docs:
        doc_storage = Document(user_id=user_id, text=doc.text, id=uuid.UUID(doc.doc_id), data_source=data_source.name)
        docs_storage.append(doc_storage)

    # use llama index to run embeddings code
    with suppress_stdout():
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=embed_model, chunk_size=config.default_embedding_config.embedding_chunk_size
        )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)
    doc_info = index.ref_doc_info
    passages = []
    doc_dict = {}
    passages_dict = index.docstore.docs

    # SimpleVectorStore has a get method and takes in the node id to retrieve the embedding.
    if isinstance(index.vector_store, SimpleVectorStore):
        simple_store = index.vector_store
        for doc_id, data in doc_info.items():
            nodes_dict = []
            for node_id in data.node_ids:
                nodes_dict.append((node_id, simple_store.get(node_id), passages_dict[node_id].to_dict()))
            doc_dict[doc_id] = nodes_dict

    for doc_id, doc_data in tqdm(doc_dict.items()):
        for data in doc_data:
            node = data[2]
            text = node["text"].replace("\x00", "\uFFFD")  # hacky fix for error on null characters
            assert (
                len(data[1]) == config.default_embedding_config.embedding_dim
            ), f"Expected embedding dimension {config.default_embedding_config.embedding_dim}, got {len(data[1])}: {data[1]}"
            passages.append(
                Passage(
                    id=uuid.UUID(data[0]),
                    user_id=user.id,
                    text=text,
                    data_source=name,
                    embedding=data[1],
                    metadata=None,
                    doc_id=uuid.UUID(doc_id),
                    embedding_dim=config.default_embedding_config.embedding_dim,
                    embedding_model=config.default_embedding_config.embedding_model,
                )
            )

    storage.insert_many(docs_storage)
    insert_passages_into_source(passages, name, user_id, config)
    storage.save()


@app.command("index")
def load_index(
    name: str = typer.Option(help="Name of dataset to load."),
    dir: str = typer.Option(help="Path to directory containing index."),
    user_id: uuid.UUID = None,
):
    """Load a LlamaIndex saved VectorIndex into MemGPT"""
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
    name: str = typer.Option(help="Name of dataset to load."),
    input_dir: str = typer.Option(None, help="Path to directory containing dataset."),
    input_files: List[str] = typer.Option(None, help="List of paths to files containing dataset."),
    recursive: bool = typer.Option(False, help="Recursively search for files in directory."),
    extensions: str = typer.Option(default_extensions, help="Comma separated list of file extensions to load"),
    user_id: str = typer.Option(None, help="User ID to associate with dataset."),
):
    try:
        from llama_index import SimpleDirectoryReader

        if recursive:
            assert input_dir is not None, "Must provide input directory if recursive is True."

        if input_dir is not None:
            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                recursive=recursive,
                required_exts=[ext.strip() for ext in extensions.split(",")],
            )
        else:
            reader = SimpleDirectoryReader(input_files=input_files)

        # load docs
        docs = []
        for data in reader.iter_data():
            doc = "".join([doc.text[2:] for doc in data])
            doco = llama_index.Document()
            doco.set_content(doc)
            docs.append(doco)
        store_docs(name, docs, user_id)

    except ValueError as e:
        typer.secho(f"Failed to load directory from provided information.\n{e}", fg=typer.colors.RED)


@app.command("webpage")
def load_webpage(
    name: str = typer.Option(help="Name of dataset to load."),
    urls: List[str] = typer.Option(None, help="List of urls to load."),
):
    try:
        from llama_index.readers.web import SimpleWebPageReader

        docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
        store_docs(name, docs)

    except ValueError as e:
        typer.secho(f"Failed to load webpage from provided information.\n{e}", fg=typer.colors.RED)


@app.command("database")
def load_database(
    name: str = typer.Option(help="Name of dataset to load."),
    query: str = typer.Option(help="Database query."),
    dump_path: str = typer.Option(None, help="Path to dump file."),
    scheme: str = typer.Option(None, help="Database scheme."),
    host: str = typer.Option(None, help="Database host."),
    port: int = typer.Option(None, help="Database port."),
    user: str = typer.Option(None, help="Database user."),
    password: str = typer.Option(None, help="Database password."),
    dbname: str = typer.Option(None, help="Database name."),
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
                port=port,  # Database Port
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
    name: str = typer.Option(help="Name of dataset to load."),
    uri: str = typer.Option(help="Database URI."),
    table_name: str = typer.Option(help="Name of table containing data."),
    text_column: str = typer.Option(help="Name of column containing text."),
    embedding_column: str = typer.Option(help="Name of column containing embedding."),
    user_id: uuid.UUID = None,
):
    """Load pre-computed embeddings into MemGPT from a database."""

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
        select_statement = select(table.c[text_column], table.c[embedding_column].cast(Vector(config.embedding_dim)))

        # Execute the query and fetch the results
        with engine.connect() as connection:
            result = connection.execute(select_statement).fetchall()

        # Convert to a list of tuples (text, embedding)
        passages = []
        for text, embedding in result:
            # assume that embeddings are the same model as in config
            passages.append(
                Passage(text=text, embedding=embedding, embedding_dim=config.embedding_dim, embedding_model=config.embedding_model)
            )
            assert config.embedding_dim == len(embedding), f"Expected embedding dimension {config.embedding_dim}, got {len(embedding)}"

        # create storage connector
        config = MemGPTConfig.load()
        if user_id is None:
            user_id = uuid.UUID(config.anon_clientid)

        insert_passages_into_source(passages, name, user_id, config)

    except ValueError as e:
        typer.secho(f"Failed to load vector database from provided information.\n{e}", fg=typer.colors.RED)
