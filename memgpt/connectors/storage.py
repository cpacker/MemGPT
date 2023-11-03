""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""
from typing import Optional, List
from memgpt.config import AgentConfig, MemGPTConfig
from tqdm import tqdm
import re
import pickle
import os

from pgvector.psycopg import register_vector
from pgvector.sqlalchemy import Vector
import psycopg


from sqlalchemy import create_engine, Column, String, Integer, LargeBinary, Table, BIGINT, select, inspect
from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import List, Optional
from abc import abstractmethod
import numpy as np
from tqdm import tqdm

from llama_index import (
    VectorStoreIndex,
    EmptyIndex,
    get_response_synthesizer,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.schema import BaseComponent, TextNode, Document


from memgpt.constants import MEMGPT_DIR


class Passage:
    """A passage is a single unit of memory, and a standard format accross all storage backends.

    It is a string of text with an associated embedding.
    """

    def __init__(self, text: str, embedding: np.ndarray, doc_id: Optional[str] = None, passage_id: Optional[str] = None):
        self.text = text
        self.embedding = embedding
        self.doc_id = doc_id
        self.passage_id = passage_id

    def __repr__(self):
        return f"Passage(text={self.text}, embedding={self.embedding})"


class StorageConnector:
    def sanitize_table_name(self, name: str) -> str:
        # Remove leading and trailing whitespace
        name = name.strip()

        # Replace spaces and invalid characters with underscores
        name = re.sub(r"\s+|\W+", "_", name)

        # SQL identifiers should not start with a number or underscore (for some databases)
        if name[0].isdigit() or name[0] == "_":
            name = "t" + name

        # Truncate to the maximum identifier length (e.g., 63 for PostgreSQL)
        max_length = 63
        if len(name) > max_length:
            name = name[:max_length].rstrip("_")

        # Convert to lowercase
        name = name.lower()

        return name

    def generate_table_name_agent(self, agent_config: AgentConfig):
        return f"memgpt_agent_{self.sanitize_table_name(agent_config.name)}"

    def generate_table_name(self, name: str):
        return f"memgpt_{self.sanitize_table_name(name)}"

    @staticmethod
    def get_storage_connector(name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        storage_type = MemGPTConfig.load().archival_storage_type
        if storage_type == "local":
            return LocalStorageConnector(name=name, agent_config=agent_config)
        elif storage_type == "postgres":
            return PostgresStorageConnector(name=name, agent_config=agent_config)
        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def list_loaded_data():
        storage_type = MemGPTConfig.load().archival_storage_type
        if storage_type == "local":
            return LocalStorageConnector.list_loaded_data()
        elif storage_type == "postgres":
            return PostgresStorageConnector.list_loaded_data()
        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @abstractmethod
    def get_all(self) -> List[Passage]:
        pass

    @abstractmethod
    def get(self, id: str) -> Passage:
        pass

    @abstractmethod
    def insert(self, passage: Passage):
        pass

    @abstractmethod
    def insert_many(self, passages: List[Passage]):
        pass

    @abstractmethod
    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        pass

    @abstractmethod
    def save(self):
        """Save state of storage connector"""
        pass


class LocalStorageConnector:

    """Local storage connector based on LlamaIndex"""

    def __init__(self, name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):

        from memgpt.embeddings import embedding_model

        config = MemGPTConfig.load()

        # TODO: add asserts to avoid both being passed
        if name is None:
            self.name = agent_config.name
            self.save_directory = agent_config.save_agent_index_dir()
        else:
            self.name = name
            self.save_directory = f"{MEMGPT_DIR}/archival/{name}"

        # llama index contexts
        self.embed_model = embedding_model(config)
        self.service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model, chunk_size=config.embedding_chunk_size)

        # load/create index
        self.save_path = f"{self.save_directory}/nodes.pkl"
        print("save", self.save_path)
        if os.path.exists(self.save_path):
            self.nodes = pickle.load(open(self.save_path, "rb"))
        else:
            self.nodes = []

        print("nodes", len(self.nodes))

        # create vectorindex
        if len(self.nodes):
            # self.storage_context = StorageContext.from_defaults(persist_dir=self.save_directory)
            # self.index = load_index_from_storage(self.storage_context)
            # llama index is trash so we just deal with nodes, and create an index from nodes
            self.index = VectorStoreIndex(self.nodes)
        else:
            self.index = EmptyIndex()

    def get_nodes(self) -> List[TextNode]:
        """Get llama index nodes"""
        embed_dict = self.index._vector_store._data.embedding_dict
        node_dict = self.index._docstore.docs

        nodes = []
        for node_id, node in node_dict.items():
            vector = embed_dict[node_id]
            node.embedding = vector
            nodes.append(TextNode(text=node.text, embedding=vector))
        return nodes

    def add_nodes(self, nodes: List[TextNode]):
        self.nodes += nodes
        self.index = VectorStoreIndex(self.nodes)

    def get_all(self) -> List[Passage]:
        passages = []
        for node in self.get_nodes():
            assert node.embedding is not None, f"Node embedding is None"
            passages.append(Passage(text=node.text, embedding=node.embedding))
        return passages

    def get(self, id: str) -> Passage:
        pass

    def insert(self, passage: Passage):
        nodes = [TextNode(text=passage.text, embedding=passage.embedding)]
        self.nodes += nodes
        if isinstance(self.index, EmptyIndex):
            self.index = VectorStoreIndex(self.nodes, service_context=self.service_context, show_progress=True)
        else:
            self.index.insert_nodes(nodes)

    def insert_many(self, passages: List[Passage]):
        nodes = [TextNode(text=passage.text, embedding=passage.embedding) for passage in passages]
        self.nodes += nodes
        if isinstance(self.index, EmptyIndex):
            self.index = VectorStoreIndex(self.nodes, service_context=self.service_context, show_progress=True)
            print("new size", len(self.get_nodes()))
        else:
            orig_size == len(self.get_nodes())
            self.index.insert_nodes(nodes)
            assert len(self.get_nodes()) == orig_size + len(
                passages
            ), f"expected {orig_size + len(passages)} nodes, got {len(self.get_nodes())} nodes"

    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        # TODO: this may be super slow?
        retriever = VectorIndexRetriever(
            index=self.index,  # does this get refreshed?
            similarity_top_k=top_k,
        )
        nodes = retriever.retrieve(query)
        results = [Passage(embedding=node.embedding, text=node.text) for node in nodes]
        print(results)
        return results

    def save(self):
        # if isinstance(self.index, EmptyIndex):
        #    print("no index to save")
        #    return
        assert len(self.nodes) == len(self.get_nodes()), f"Expected {len(self.nodes)} nodes, got {len(self.get_nodes())} nodes"
        os.makedirs(self.save_directory, exist_ok=True)
        pickle.dump(self.nodes, open(self.save_path, "wb"))
        print("saved", self.save_path)

    @staticmethod
    def list_loaded_data():
        sources = []
        for data_source_file in os.listdir(os.path.join(MEMGPT_DIR, "archival")):
            name = os.path.basename(data_source_file)
            sources.append(name)
        return sources


Base = declarative_base()

# Define the SQLAlchemy ORM model for the Passage table


class PassageModel(Base):
    # __tablename__ = 'test2'
    __abstract__ = True  # this line is necessary

    # Assuming passage_id is the primary key
    id = Column(BIGINT, primary_key=True, nullable=False, autoincrement=True)
    doc_id = Column(String)
    text = Column(String, nullable=False)
    embedding = mapped_column(Vector(1536))  # TODO: don't hard-code
    # metadata_ = Column(JSON(astext_type=Text()))

    def __repr__(self):
        return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"


def get_db_model(table_name: str):
    class_name = f"{table_name.capitalize()}Model"
    Model = type(class_name, (PassageModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
    return Model


class PostgresStorageConnector(StorageConnector):
    def __init__(self, name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        config = MemGPTConfig.load()

        # determine table name
        if agent_config:
            assert name is None, f"Cannot specify both agent config and name {name}"
            self.table_name = self.generate_table_name_agent(agent_config)
        elif name:
            assert agent_config is None, f"Cannot specify both agent config and name {name}"
            self.table_name = self.generate_table_name(name)
        else:
            raise ValueError("Must specify either agent config or name")

        print("table name", self.table_name)

        # create table
        self.uri = config.archival_storage_uri
        self.db_model = get_db_model(self.table_name)
        self.engine = create_engine(self.uri)
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)

    def get_all(self) -> List[Passage]:
        session = self.Session()
        db_passages = session.query(self.db_model).all()
        return [Passage(text=p.text, embedding=p.embedding, doc_id=p.doc_id, passage_id=p.id) for p in db_passages]

    def get(self, id: str) -> Optional[Passage]:
        session = self.Session()
        db_passage = session.query(self.db_model).get(id)
        if db_passage is None:
            return None
        return Passage(text=db_passage.text, embedding=db_passage.embedding, doc_id=db_passage.doc_id, passage_id=db_passage.passage_id)

    def insert(self, passage: Passage):
        session = self.Session()
        db_passage = self.db_model(doc_id=passage.doc_id, text=passage.text, embedding=passage.embedding)
        session.add(db_passage)
        session.commit()

    def insert_many(self, passages: List[Passage], show_progress=True):
        session = self.Session()
        iterable = tqdm(passages) if show_progress else passages
        for passage in iterable:
            db_passage = self.db_model(doc_id=passage.doc_id, text=passage.text, embedding=passage.embedding)
            session.add(db_passage)
        session.commit()

    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        session = self.Session()
        # Assuming PassageModel.embedding has the capability of computing l2_distance
        results = session.scalars(select(self.db_model).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)).all()

        # Convert the results into Passage objects
        passages = [
            Passage(text=result.text, embedding=np.frombuffer(result.embedding), doc_id=result.doc_id, passage_id=result.id)
            for result in results
        ]
        print(passages[0].text)

        return passages

    def delete(self):
        """Drop the passage table from the database."""
        # Bind the engine to the metadata of the base class so that the
        # declaratives can be accessed through a DBSession instance
        Base.metadata.bind = self.engine

        # Drop the table specified by the PassageModel class
        self.db_model.__table__.drop(self.engine)

    def save(self):
        # don't need to save
        print("Saving db")
        return

    @staticmethod
    def list_loaded_data():
        config = MemGPTConfig.load()
        engine = create_engine(config.archival_storage_uri)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        tables = [table for table in tables if table.startswith("memgpt_") and not table.startswith("memgpt_agent_")]
        tables = [table.replace("memgpt_", "") for table in tables]
        return tables
