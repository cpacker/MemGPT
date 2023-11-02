""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""
from typing import Optional, List
from memgpt.config import AgentConfig, MemGPTConfig
from tqdm import tqdm
import re

from pgvector.psycopg import register_vector
from pgvector.sqlalchemy import Vector
import psycopg


from sqlalchemy import create_engine, Column, String, Integer, LargeBinary, Table, BIGINT, select
from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import List, Optional
from abc import abstractmethod
import numpy as np
from tqdm import tqdm


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
    def query(self, query_string: str, top_k: int = 10) -> List[Passage]:
        pass

    @abstractmethod
    def save(self):
        """Save state of storage connector"""
        pass

    @abstractmethod
    def list_tables(self):
        """List all tables in the storage connector"""
        pass


class LocalStorageConnector:

    """Local storage connector based on LlamaIndex"""

    def __init__(self, agent_config: AgentConfig):
        pass

    def get_all(self) -> List[Passage]:
        pass

    def get(self, id: str) -> Passage:
        pass

    def insert(self, passage: Passage):
        pass

    def insert_many(self, passages: List[Passage]):
        pass

    def query(self, query_string: str, top_k: int = 10) -> List[Passage]:
        pass

    def save(self):
        """Save state of storage connector"""
        pass


# class PostgresStorageConnector:
#
#
#
#
#
#    def __init__(self, agent_config: AgentConfig, uri):
#
#        config = MemGPTConfig.load()
#        self.table_name = self.table_name(agent_config)
#        self.uri = uri
#
#        # create table
#        # TODO: fix
#        self.conn = psycopg.connect(dbname='pgvector_example', autocommit=True)
#        self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
#        register_vector(self.conn)
#        #self.conn.execute('DROP TABLE IF EXISTS documents') # TODO: don't do this!
#
#        # check if already exists
#        self.conn.execute(f'CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector({self.config.embedding_dim}))')
#
#
#    @abstractmethod
#    def get_all(self) -> List[Passage]:
#        pass
#
#    @abstractmethod
#    def get(self, id: str) -> Passage:
#        pass
#
#    def insert(self, passage: Passage):
#        self.conn.execute(f'INSERT INTO documents (content, embedding) VALUES (%s, %s)', (passage.text, passage.embedding))
#
#    def insert_many(self, passages: List[Passage], show_progress=True):
#        if show_progress:
#            for passage in tqdm(passages):
#                self.insert(passage)
#        else:
#            for passage in passages:
#                self.insert(passage)
#
#    def query(self, query_vector: List[float], top_k: int = 10) -> List[Passage]:
#        results = self.conn.execute(f'SELECT * FROM item ORDER BY embedding <-> %s LIMIT {top_k}', (query_vector,)).fetchall()
#        # TODO: convert to passages
#
#    @abstractmethod
#    def save(self):
#        """ Save state of storage connector """
#        pass


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

    def query(self, query_vector: List[float], top_k: int = 10) -> List[Passage]:
        session = self.Session()
        # Assuming PassageModel.embedding has the capability of computing l2_distance
        results = session.scalars(select(self.db_model).order_by(self.db_model.embedding.l2_distance(query_vector)).limit(top_k)).all()

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
        # Since SQLAlchemy commits changes individually in `insert` and `insert_many`, this might not be needed.
        # If there's a need to handle transactions manually, you can control them using the session object.
        pass
