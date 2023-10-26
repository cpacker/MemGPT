""" 
This file contains functions for loading data into MemGPT's archival storage.

Data can be loaded with the following command, once a load function is defined: 
```
memgpt load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""

from llama_index import download_loader
from llama_index.embeddings import OpenAIEmbedding
from typing import List
import os
import typer
from memgpt.constants import MEMGPT_DIR
from memgpt.utils import estimate_openai_cost, index_docs, save_index

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
        reader = SimpleDirectoryReader(
            input_files=input_files
        )

    # load docs
    print("Loading data...")
    docs = reader.load_data()

    # embed docs 
    print("Indexing documents...")
    index = index_docs(docs)
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
    index = index_docs(docs)
    # save connector information into .memgpt metadata file
    save_index(index, name)

