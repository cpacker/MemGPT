from memgpt.data_types import Passage, Document, EmbeddingConfig, Source
from memgpt.utils import create_uuid_from_string
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.embeddings import embedding_model
from memgpt.data_types import Document, Passage

import uuid
from typing import List, Iterator, Dict, Tuple, Optional
from llama_index.core import Document as LlamaIndexDocument


class DataConnector:
    def __init__(self, data_source):
        self.data_source = data_source

    def generate_documents(self) -> Iterator[Document]:
        pass

    def generate_passages(self, documents: List[Document]) -> Iterator[Passage]:
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


class DirectoryConnector:
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
