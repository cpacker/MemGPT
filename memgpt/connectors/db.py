from pgvector.psycopg import register_vector
from pgvector.sqlalchemy import Vector
import psycopg


from sqlalchemy import create_engine, Column, String, BIGINT, select, inspect, text
from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

import re
from tqdm import tqdm
from typing import Optional, List, Iterator
import numpy as np
from tqdm import tqdm
import pandas as pd

from memgpt.config import MemGPTConfig
from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.config import AgentConfig, MemGPTConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.utils import printd

Base = declarative_base()


def get_db_model(table_name: str):
    config = MemGPTConfig.load()

    class PassageModel(Base):
        """Defines data model for storing Passages (consisting of text, embedding)"""

        __abstract__ = True  # this line is necessary

        # Assuming passage_id is the primary key
        id = Column(BIGINT, primary_key=True, nullable=False, autoincrement=True)
        doc_id = Column(String)
        text = Column(String, nullable=False)
        embedding = mapped_column(Vector(config.embedding_dim))
        # metadata_ = Column(JSON(astext_type=Text()))

        def __repr__(self):
            return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

    """Create database model for table_name"""
    class_name = f"{table_name.capitalize()}Model"
    Model = type(class_name, (PassageModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
    return Model


class PostgresStorageConnector(StorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

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

        printd(f"Using table name {self.table_name}")

        # create table
        self.uri = config.archival_storage_uri
        if config.archival_storage_uri is None:
            raise ValueError(f"Must specifiy archival_storage_uri in config {config.config_path}")
        self.db_model = get_db_model(self.table_name)
        self.engine = create_engine(self.uri)
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)
        self.Session().execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def get_all_paginated(self, page_size: int) -> Iterator[List[Passage]]:
        session = self.Session()
        offset = 0
        while True:
            # Retrieve a chunk of records with the given page_size
            db_passages_chunk = session.query(self.db_model).offset(offset).limit(page_size).all()

            # If the chunk is empty, we've retrieved all records
            if not db_passages_chunk:
                break

            # Yield a list of Passage objects converted from the chunk
            yield [Passage(text=p.text, embedding=p.embedding, doc_id=p.doc_id, passage_id=p.id) for p in db_passages_chunk]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self, limit=10) -> List[Passage]:
        session = self.Session()
        db_passages = session.query(self.db_model).limit(limit).all()
        return [Passage(text=p.text, embedding=p.embedding, doc_id=p.doc_id, passage_id=p.id) for p in db_passages]

    def get(self, id: str) -> Optional[Passage]:
        session = self.Session()
        db_passage = session.query(self.db_model).get(id)
        if db_passage is None:
            return None
        return Passage(text=db_passage.text, embedding=db_passage.embedding, doc_id=db_passage.doc_id, passage_id=db_passage.passage_id)

    def size(self) -> int:
        # return size of table
        session = self.Session()
        return session.query(self.db_model).count()

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
        return passages

    def delete(self):
        """Drop the passage table from the database."""
        # Bind the engine to the metadata of the base class so that the
        # declaratives can be accessed through a DBSession instance
        Base.metadata.bind = self.engine

        # Drop the table specified by the PassageModel class
        self.db_model.__table__.drop(self.engine)

    def save(self):
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

    def sanitize_table_name(self, name: str) -> str:
        # Remove leading and trailing whitespace
        name = name.strip()

        # Replace spaces and invalid characters with underscores
        name = re.sub(r"\s+|\W+", "_", name)

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


class LanceDBConnector(StorageConnector):
    """Storage via LanceDB"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, name: Optional[str] = None):
        config = MemGPTConfig.load()

        # determine table name
        if name:
            self.table_name = self.generate_table_name(name)
        else:
            self.table_name = "lancedb_tbl"

        printd(f"Using table name {self.table_name}")

        # create table
        self.uri = config.archival_storage_uri
        if config.archival_storage_uri is None:
            raise ValueError(f"Must specifiy archival_storage_uri in config {config.config_path}")
        import lancedb

        self.db = lancedb.connect(self.uri)
        self.table = None

    def get_all_paginated(self, page_size: int) -> Iterator[List[Passage]]:
        session = self.Session()
        offset = 0
        while True:
            # Retrieve a chunk of records with the given page_size
            db_passages_chunk = self.table.search().limit(page_size).to_list()

            # If the chunk is empty, we've retrieved all records
            if not db_passages_chunk:
                break

            # Yield a list of Passage objects converted from the chunk
            yield [
                Passage(text=p["text"], embedding=p["vector"], doc_id=p["doc_id"], passage_id=p["passage_id"]) for p in db_passages_chunk
            ]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self, limit=10) -> List[Passage]:
        db_passages = self.table.search().limit(limit).to_list()
        return [Passage(text=p["text"], embedding=p["vector"], doc_id=p["doc_id"], passage_id=p["passage_id"]) for p in db_passages]

    def get(self, id: str) -> Optional[Passage]:
        db_passage = self.table.where(f"passage_id={id}").to_list()
        if len(db_passage) == 0:
            return None
        return Passage(
            text=db_passage["text"], embedding=db_passage["embedding"], doc_id=db_passage["doc_id"], passage_id=db_passage["passage_id"]
        )

    def size(self) -> int:
        # return size of table
        if self.table:
            return len(self.table.search().to_list())
        else:
            print(f"Table with name {self.table_name} not present")
            return 0

    def insert(self, passage: Passage):
        data = [{"doc_id": passage.doc_id, "text": passage.text, "passage_id": passage.passage_id, "vector": passage.embedding}]

        if self.table:
            self.table.add(data)
        else:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")

    def insert_many(self, passages: List[Passage], show_progress=True):
        data = []
        iterable = tqdm(passages) if show_progress else passages
        for passage in iterable:
            temp_dict = {"doc_id": passage.doc_id, "text": passage.text, "passage_id": passage.passage_id, "vector": passage.embedding}
            data.append(temp_dict)

        if self.table:
            self.table.add(data)
        else:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")

    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        # Assuming query_vec is of same length as embeddings inside table
        results = self.table.search(query_vec).limit(top_k)

        # Convert the results into Passage objects
        passages = [
            Passage(text=result["text"], embedding=result["embedding"], doc_id=result["doc_id"], passage_id=result["passage_id"])
            for result in results
        ]
        return passages

    def delete(self):
        """Drop the passage table from the database."""
        # Drop the table specified by the PassageModel class
        self.db.drop_table(self.table_name)

    def save(self):
        return

    @staticmethod
    def list_loaded_data():
        config = MemGPTConfig.load()
        import lancedb

        db = lancedb.connect(config.archival_storage_uri)

        tables = db.table_names()
        tables = [table for table in tables if table.startswith("memgpt_")]
        tables = [table.replace("memgpt_", "") for table in tables]
        return tables

    def sanitize_table_name(self, name: str) -> str:
        # Remove leading and trailing whitespace
        name = name.strip()

        # Replace spaces and invalid characters with underscores
        name = re.sub(r"\s+|\W+", "_", name)

        # Truncate to the maximum identifier length
        max_length = 63
        if len(name) > max_length:
            name = name[:max_length].rstrip("_")

        # Convert to lowercase
        name = name.lower()

        return name

    def generate_table_name(self, name: str):
        return f"memgpt_{self.sanitize_table_name(name)}"
