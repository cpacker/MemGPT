from memgpt.data_sources.connectors import DataConnector
from memgpt.agent import Agent
from memgpt.data_types import Passage, Document, EmbeddingConfig, Source
from memgpt.utils import create_uuid_from_string
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.embeddings import embedding_model
from memgpt.data_types import Document, Passage

import uuid
from typing import List, Iterator
from llama_index.core import Document as LlamaIndexDocument


class DataConnector:
    def __init__(self, data_source):
        self.data_source = data_source

    def generate_documents(self) -> Iterator[Document]:
        pass

    def generate_passages(self, documents: List[Document]) -> Iterator[Passage]:
        pass


def load_data(
    self,
    connector: DataConnector,
    source: Source,
    embedding_config: EmbeddingConfig,
    document_store: StorageConnector,
    passage_store: StorageConnector,
    chunk_size: int = 1000,
):
    """Load data from a connector (generates documents and passages) into a specified source_id, associatedw with a user_id."""

    # embedding model
    embed_model = embedding_model(embedding_config)

    # insert passages/documents
    passages = []
    passage_count = 0
    document_count = 0
    for document in connector.generate_documents():
        # insert document into storage
        document.user_id = source.user_id
        document.id = create_uuid_from_string(f"{str(source.id)}_{document.text}")
        document_count += 1
        document_store.insert(document)

        # generate passages
        for passage in connector.generate_passages([document]):
            passage.id = create_uuid_from_string(f"{str(source.id)}_{passage.text}")
            passage.user_id = source.user_id
            passage.embedding_dim = embedding_config.embedding_dim
            passage.embedding_model = embedding_config.embedding_model

            # compute passage embeddings
            passage.embedding = embed_model.get_text_embedding(passage.text)

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


class DirectoryConnector:
    def __init__(self, input_files: List[str] = None, input_directory: str = None, recursive: bool = False):
        self.connector_type = "directory"
        self.input_files = input_files
        self.input_directory = input_directory
        self.recursive = recursive
        self.extensions = None  # TODO: fix

        if self.recursive == True:
            assert self.input_dir is not None, "Must provide input directory if recursive is True."

    def generate_documents(self) -> Iterator[Document]:

        from llama_index import SimpleDirectoryReader

        if self.input_dir is not None:
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
            doc = Document(text=llama_index_doc.text, metadata=llama_index_doc.metadata)
            docs.append(doc)

    def generate_passages(self, documents: List[Document], chunk_size: int = 1024) -> Iterator[Passage]:
        # use llama index to run embeddings code
        from llama_index.core.node_parser import SentenceSplitter

        parser = SentenceSplitter(chunk_size=chunk_size)
        for document in documents:
            llama_index_docs = [LlamaIndexDocument(text=document.text, metadata=document.metadata)]
            nodes = parser.get_nodes_from_documents(llama_index_docs)
            for node in nodes:
                passage = Passage(
                    text=node.text,
                    doc_id=document.id,
                )
                yield passage
