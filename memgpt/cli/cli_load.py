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
from memgpt.embeddings import Index, embedding_model
from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.config import MemGPTConfig

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
)

app = typer.Typer()


def store_docs(name, docs, show_progress=True):
    """Common function for embedding and storing documents"""
    storage = StorageConnector.get_storage_connector(name=name)
    config = MemGPTConfig.load()
    embed_model = embedding_model(config)

    # use llama index to run embeddings code
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=config.embedding_chunk_size)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
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
        ), f"Expected embedding dimension {config.embedding_dim}, got {len(node.embedding)}"
        passages.append(Passage(text=text, embedding=vector))

    # embeddings = [embed_model.get_text_embedding(doc.text) for doc in docs]
    # storage.insert_many([Passage(text=doc.text, embedding=embedding) for doc, embedding in zip(docs, embeddings)])

    # insert into storage
    storage.insert_many(passages)


@app.command("index")
def load_index(
    name: str = typer.Option(help="Name of dataset to load."), dir: str = typer.Option(help="Path to directory containing index.")
):
    """Load a LlamaIndex saved VectorIndex into MemGPT"""
    from llama_index import load_index_from_storage, StorageContext

    # load index data
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    loaded_index = load_index_from_storage(storage_context)

    embed_dict = loaded_index._vector_store._data.embedding_dict
    node_dict = loaded_index._docstore.docs

    passages = []
    for node_id, node in node_dict.items():
        vector = embed_dict[node_id]
        node.embedding = vector
        passages.append(Passage(text=node.text, embedding=vector))

    # index = Index(name)
    # index.load_nodes(nodes)

    storage = StorageConnector.get_storage_connector(name=name)
    storage.insert_many(passages)


@app.command("directory")
def load_directory(
    name: str = typer.Option(help="Name of dataset to load."),
    input_dir: str = typer.Option(None, help="Path to directory containing dataset."),
    input_files: List[str] = typer.Option(None, help="List of paths to files containing dataset."),
    recursive: bool = typer.Option(False, help="Recursively search for files in directory."),
):
    from llama_index import SimpleDirectoryReader
    from memgpt.utils import get_index, save_index

    if recursive:
        assert input_dir is not None, "Must provide input directory if recursive is True."

    if input_dir is not None:
        assert len(input_files) == 0, "Either load in a list of files OR a directory."
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            recursive=recursive,
        )
    else:
        reader = SimpleDirectoryReader(input_files=input_files)

    # load docs
    print("Loading data...")
    docs = reader.load_data()

    store_docs(name, docs)

    # index = Index(name)
    # index.load_documents(docs)

    ## embed docs
    # print("Indexing documents...")
    # index = get_index(name, docs)
    ## save connector information into .memgpt metadata file
    # save_index(index, name)


@app.command("webpage")
def load_webpage(
    name: str = typer.Option(help="Name of dataset to load."),
    urls: List[str] = typer.Option(None, help="List of urls to load."),
):
    from llama_index import SimpleWebPageReader
    from memgpt.utils import get_index, save_index

    docs = SimpleWebPageReader(html_to_text=True).load_data(urls)

    # embed docs
    print("Indexing documents...")
    index = get_index(docs)
    # save connector information into .memgpt metadata file
    save_index(index, name)


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
    from memgpt.utils import get_index, save_index

    print(dump_path, scheme)

    if dump_path is not None:
        # read from database dump file
        from sqlalchemy import create_engine, MetaData

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

    index = get_index(name, docs)
    save_index(index, name)
