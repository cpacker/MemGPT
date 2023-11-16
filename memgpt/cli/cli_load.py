"""
This file contains functions for loading data into MemGPT's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
memgpt load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

from typing import List
from tqdm import tqdm
import typer
from memgpt.embeddings import embedding_model
from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.config import MemGPTConfig

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

app = typer.Typer()


def store_docs(name, docs, show_progress=True):
    """Common function for embedding and storing documents"""

    storage = StorageConnector.get_storage_connector(name=name)
    config = MemGPTConfig.load()
    embed_model = embedding_model()

    # use llama index to run embeddings code
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=config.embedding_chunk_size)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)
    embed_dict = index._vector_store._data.embedding_dict
    node_dict = index._docstore.docs

    # gather passages
    passages = []
    for node_id, node in tqdm(node_dict.items()):
        vector = embed_dict[node_id]
        node.embedding = vector
        text = node.text.replace("\x00", "\uFFFD")  # hacky fix for error on null characters
        assert (
            len(node.embedding) == config.embedding_dim
        ), f"Expected embedding dimension {config.embedding_dim}, got {len(node.embedding)}: {node.embedding}"
        passages.append(Passage(text=text, embedding=vector))

    # insert into storage
    storage.insert_many(passages)
    storage.save()


@app.command("index")
def load_index(
    name: str = typer.Option(help="Name of dataset to load."), dir: str = typer.Option(help="Path to directory containing index.")
):
    """Load a LlamaIndex saved VectorIndex into MemGPT"""
    # load index data
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    loaded_index = load_index_from_storage(storage_context)

    # hacky code to extract out passages/embeddings (thanks a lot, llama index)
    embed_dict = loaded_index._vector_store._data.embedding_dict
    node_dict = loaded_index._docstore.docs

    passages = []
    for node_id, node in node_dict.items():
        vector = embed_dict[node_id]
        node.embedding = vector
        passages.append(Passage(text=node.text, embedding=vector))

    # create storage connector
    storage = StorageConnector.get_storage_connector(name=name)

    # add and save all passages
    storage.insert_many(passages)
    storage.save()


@app.command("directory")
def load_directory(
    name: str = typer.Option(help="Name of dataset to load."),
    input_dir: str = typer.Option(None, help="Path to directory containing dataset."),
    input_files: List[str] = typer.Option(None, help="List of paths to files containing dataset."),
    recursive: bool = typer.Option(False, help="Recursively search for files in directory."),
):
    from llama_index import SimpleDirectoryReader

    if recursive:
        assert input_dir is not None, "Must provide input directory if recursive is True."

    if input_dir is not None:
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            recursive=recursive,
        )
    else:
        reader = SimpleDirectoryReader(input_files=input_files)

    # load docs
    print("loading data")
    docs = reader.load_data()
    print("done loading data")
    store_docs(name, docs)


@app.command("webpage")
def load_webpage(
    name: str = typer.Option(help="Name of dataset to load."),
    urls: List[str] = typer.Option(None, help="List of urls to load."),
):
    from llama_index import SimpleWebPageReader

    docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
    store_docs(name, docs)


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


@app.command("vector-database")
def load_vector_database(
    name: str = typer.Option(help="Name of dataset to load."),
    uri: str = typer.Option(help="Database URI."),
    table_name: str = typer.Option(help="Name of table containing data."),
    text_column: str = typer.Option(help="Name of column containing text."),
    embedding_column: str = typer.Option(help="Name of column containing embedding."),
):
    """Load pre-computed embeddings into MemGPT from a database."""

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
        passages.append(Passage(text=text, embedding=embedding))
        assert config.embedding_dim == len(embedding), f"Expected embedding dimension {config.embedding_dim}, got {len(embedding)}"

    # insert into storage
    storage = StorageConnector.get_storage_connector(name=name)
    storage.insert_many(passages)
