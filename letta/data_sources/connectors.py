from typing import Dict, Iterator, List, Tuple

import typer
from llama_index.core import Document as LlamaIndexDocument
from llama_index.readers.file import PDFReader

from letta.agent_store.storage import StorageConnector
from letta.embeddings import embedding_model
from letta.schemas.file import File
from letta.schemas.passage import Passage
from letta.schemas.source import Source
from letta.utils import create_uuid_from_string


class DataConnector:
    """
    Base class for data connectors that can be extended to generate files and passages from a custom data source.
    """

    def generate_files(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[File]:
        """
        Generate document text and metadata from a data source.

        Returns:
            files (Iterator[Tuple[str, Dict]]): Generate a tuple of string text and metadata dictionary for each document.
        """

    def generate_passages(self, file_text: str, file: File, chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        """
        Generate passage text and metadata from a list of files.

        Args:
            file_text (str): The text of the document
            file (File): The document to generate passages from.
            chunk_size (int, optional): Chunk size for splitting passages. Defaults to 1024.

        Returns:
            passages (Iterator[Tuple[str, Dict]]): Generate a tuple of string text and metadata dictionary for each passage.
        """


def load_data(
    connector: DataConnector,
    source: Source,
    passage_store: StorageConnector,
    document_store: StorageConnector,
):
    """Load data from a connector (generates file and passages) into a specified source_id, associatedw with a user_id."""
    embedding_config = source.embedding_config

    # embedding model
    embed_model = embedding_model(embedding_config)

    # insert passages/file
    passages = []
    embedding_to_document_name = {}
    passage_count = 0
    document_count = 0
    for file_text, document_metadata in connector.generate_files():
        # insert document into storage
        file = File(
            user_id=source.user_id,
            source_id=source.id,
            metadata_=document_metadata,
        )
        document_count += 1
        document_store.insert(file)

        # generate passages
        for passage_text, passage_metadata in connector.generate_passages(
            file_text, file, chunk_size=embedding_config.embedding_chunk_size
        ):
            # for some reason, llama index parsers sometimes return empty strings
            if len(passage_text) == 0:
                typer.secho(
                    f"Warning: Llama index parser returned empty string, skipping insert of passage with metadata '{passage_metadata}' into VectorDB. You can usually ignore this warning.",
                    fg=typer.colors.YELLOW,
                )
                continue

            # get embedding
            try:
                embedding = embed_model.get_text_embedding(passage_text)
            except Exception as e:
                typer.secho(
                    f"Warning: Failed to get embedding for {passage_text} (error: {str(e)}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                )
                continue

            passage = Passage(
                id=create_uuid_from_string(f"{str(source.id)}_{passage_text}"),
                text=passage_text,
                file_id=file.id,
                source_id=source.id,
                metadata_=passage_metadata,
                user_id=source.user_id,
                embedding_config=source.embedding_config,
                embedding=embedding,
            )

            hashable_embedding = tuple(passage.embedding)
            file_name = file.metadata_.get("file_path", file.id)
            if hashable_embedding in embedding_to_document_name:
                typer.secho(
                    f"Warning: Duplicate embedding found for passage in {file_name} (already exists in {embedding_to_document_name[hashable_embedding]}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                )
                continue

            passages.append(passage)
            embedding_to_document_name[hashable_embedding] = file_name
            if len(passages) >= 100:
                # insert passages into passage store
                passage_store.insert_many(passages)

                passage_count += len(passages)
                passages = []

    if len(passages) > 0:
        # insert passages into passage store
        passage_store.insert_many(passages)
        passage_count += len(passages)

    return passage_count, document_count


class DirectoryConnector(DataConnector):
    def __init__(self, input_files: List[str] = None, input_directory: str = None, recursive: bool = False, extensions: List[str] = None):
        """
        Connector for reading text data from a directory of files.

        Args:
            input_files (List[str], optional): List of file paths to read. Defaults to None.
            input_directory (str, optional): Directory to read files from. Defaults to None.
            recursive (bool, optional): Whether to read files recursively from the input directory. Defaults to False.
            extensions (List[str], optional): List of file extensions to read. Defaults to None.
        """
        self.connector_type = "directory"
        self.input_files = input_files
        self.input_directory = input_directory
        self.recursive = recursive
        self.extensions = extensions

        if self.recursive == True:
            assert self.input_directory is not None, "Must provide input directory if recursive is True."

    def generate_files(self) -> Iterator[Tuple[str, Dict]]:
        from llama_index.core import SimpleDirectoryReader

        # We need to hijack the file extractor here
        # The default behavior is to split up the PDF by page, but we want to return the full document
        file_extractor = {
            ".pdf": PDFReader(return_full_document=True),
        }

        if self.input_directory is not None:
            reader = SimpleDirectoryReader(
                input_dir=self.input_directory,
                recursive=self.recursive,
                required_exts=[ext.strip() for ext in str(self.extensions).split(",")],
                exclude=["*png", "*jpg", "*jpeg"],  # Don't support images for now
                file_extractor=file_extractor,
            )
        else:
            assert self.input_files is not None, "Must provide input files if input_dir is None"
            reader = SimpleDirectoryReader(
                input_files=[str(f) for f in self.input_files],
                exclude=["*png", "*jpg", "*jpeg"],  # Don't support images for now
                file_extractor=file_extractor,
            )

        llama_index_docs = reader.load_data(show_progress=True)
        for llama_index_doc in llama_index_docs:
            yield llama_index_doc.text, llama_index_doc.metadata

    def generate_passages(self, file_text: str, file: File, chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        # use llama index to run embeddings code
        from llama_index.core.node_parser import TokenTextSplitter

        parser = TokenTextSplitter(chunk_size=chunk_size)
        llama_index_docs = [LlamaIndexDocument(text=file_text, metadata=file.metadata_)]
        nodes = parser.get_nodes_from_documents(llama_index_docs)
        for node in nodes:
            yield node.text, None


class WebConnector(DirectoryConnector):
    def __init__(self, urls: List[str] = None, html_to_text: bool = True):
        self.urls = urls
        self.html_to_text = html_to_text

    def generate_files(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Document]:
        from llama_index.readers.web import SimpleWebPageReader

        files = SimpleWebPageReader(html_to_text=self.html_to_text).load_data(self.urls)
        for document in files:
            yield document.text, {"url": document.id_}


class VectorDBConnector(DataConnector):
    # NOTE: this class has not been properly tested, so is unlikely to work
    # TODO: allow loading multiple tables (1:1 mapping between File and Table)

    def __init__(
        self,
        name: str,
        uri: str,
        table_name: str,
        text_column: str,
        embedding_column: str,
        embedding_dim: int,
    ):
        self.name = name
        self.uri = uri
        self.table_name = table_name
        self.text_column = text_column
        self.embedding_column = embedding_column
        self.embedding_dim = embedding_dim

        # connect to db table
        from sqlalchemy import create_engine

        self.engine = create_engine(uri)

    def generate_files(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Document]:
        yield self.table_name, None

    def generate_passages(self, file_text: str, file: File, chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        from pgvector.sqlalchemy import Vector
        from sqlalchemy import Inspector, MetaData, Table, select

        metadata = MetaData()
        # Create an inspector to inspect the database
        inspector = Inspector.from_engine(self.engine)
        table_names = inspector.get_table_names()
        assert self.table_name in table_names, f"Table {self.table_name} not found in database: tables that exist {table_names}."

        table = Table(self.table_name, metadata, autoload_with=self.engine)

        # Prepare a select statement
        select_statement = select(table.c[self.text_column], table.c[self.embedding_column].cast(Vector(self.embedding_dim)))

        # Execute the query and fetch the results
        # TODO: paginate results
        with self.engine.connect() as connection:
            result = connection.execute(select_statement).fetchall()

        for text, embedding in result:
            # assume that embeddings are the same model as in config
            # TODO: don't re-compute embedding
            yield text, {"embedding": embedding}
