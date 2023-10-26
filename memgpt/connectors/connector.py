"""
This file contains functions for loading data into MemGPT's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
memgpt load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

from llama_index import download_loader
from typing import List
import os
import typer
from memgpt.constants import MEMGPT_DIR
from memgpt.utils import estimate_openai_cost, get_index, save_index

app = typer.Typer()


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
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            recursive=True,
        )
    else:
        reader = SimpleDirectoryReader(input_files=input_files)

    # load docs
    print("Loading data...")
    docs = reader.load_data()

    # embed docs
    print("Indexing documents...")
    index = get_index(name, docs)
    # save connector information into .memgpt metadata file
    save_index(index, name)


@app.command("webpage")
def load_webpage(
    name: str = typer.Option(help="Name of dataset to load."),
    urls: List[str] = typer.Option(None, help="List of urls to load."),
):
    from llama_index import SimpleWebPageReader

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
