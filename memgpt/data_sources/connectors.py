from memgpt.data_types import Passage, Document, EmbeddingConfig, Source
from memgpt.utils import create_uuid_from_string
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.embeddings import embedding_model
from memgpt.data_types import Document, Passage

import uuid
from typing import List, Iterator, Dict, Tuple, Optional
from llama_index.core import Document as LlamaIndexDocument


class DataConnector:
    def generate_documents(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Document]:
        pass

    def generate_passages(self, documents: List[Document], chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        pass


def load_data(
    connector: DataConnector,
    source: Source,
    embedding_config: EmbeddingConfig,
    passage_store: StorageConnector,
    document_store: Optional[StorageConnector] = None,
    chunk_size: int = 1000,
):
    """Load data from a connector (generates documents and passages) into a specified source_id, associatedw with a user_id."""

    # embedding model
    embed_model = embedding_model(embedding_config)

    # insert passages/documents
    passages = []
    passage_count = 0
    document_count = 0
    for document_text, document_metadata in connector.generate_documents():
        # insert document into storage
        document = Document(
            id=create_uuid_from_string(f"{str(source.id)}_{document_text}"),
            text=document_text,
            metadata=document_metadata,
            data_source=source.name,
            user_id=source.user_id,
        )
        document_count += 1
        if document_store:
            document_store.insert(document)

        # generate passages
        for passage_text, passage_metadata in connector.generate_passages([document]):
            print("passage", passage_text, passage_metadata)
            embedding = embed_model.get_text_embedding(passage_text)
            passage = Passage(
                id=create_uuid_from_string(f"{str(source.id)}_{passage_text}"),
                text=passage_text,
                doc_id=document.id,
                metadata_=passage_metadata,
                user_id=source.user_id,
                data_source=source.name,
                embedding_dim=embedding_config.embedding_dim,
                embedding_model=embedding_config.embedding_model,
                embedding=embedding,
            )

            passages.append(passage)
            if len(passages) >= chunk_size:
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
        self.connector_type = "directory"
        self.input_files = input_files
        self.input_directory = input_directory
        self.recursive = recursive
        self.extensions = extensions

        if self.recursive == True:
            assert self.input_dir is not None, "Must provide input directory if recursive is True."

    def generate_documents(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Document]:
        from llama_index.core import SimpleDirectoryReader

        if self.input_directory is not None:
            reader = SimpleDirectoryReader(
                input_dir=self.input_directory,
                recursive=self.recursive,
                required_exts=[ext.strip() for ext in str(self.extensions).split(",")],
            )
        else:
            assert self.input_files is not None, "Must provide input files if input_dir is None"
            reader = SimpleDirectoryReader(input_files=[str(f) for f in self.input_files])

        llama_index_docs = reader.load_data()
        docs = []
        for llama_index_doc in llama_index_docs:
            # TODO: add additional metadata?
            # doc = Document(text=llama_index_doc.text, metadata=llama_index_doc.metadata)
            # docs.append(doc)
            yield llama_index_doc.text, llama_index_doc.metadata

    def generate_passages(self, documents: List[Document], chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        # use llama index to run embeddings code
        from llama_index.core.node_parser import SentenceSplitter

        parser = SentenceSplitter(chunk_size=chunk_size)
        for document in documents:
            llama_index_docs = [LlamaIndexDocument(text=document.text, metadata=document.metadata)]
            nodes = parser.get_nodes_from_documents(llama_index_docs)
            for node in nodes:
                # passage = Passage(
                #    text=node.text,
                #    doc_id=document.id,
                # )
                yield node.text, None


class WebConnector(DataConnector):
    # TODO

    def __init__(self):
        pass

    def generate_documents(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Document]:
        pass

    def generate_passages(self, documents: List[Document], chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        pass


class VectorDBConnector(DataConnector):
    # NOTE: this class has not been properly tested, so is unlikely to work
    # TODO: allow loading multiple tables (1:1 mapping between Document and Table)

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

    def generate_documents(self) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Document]:
        yield self.table_name, None

    def generate_passages(self, documents: List[Document], chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        from sqlalchemy import select, MetaData, Table, Inspector
        from pgvector.sqlalchemy import Vector

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
