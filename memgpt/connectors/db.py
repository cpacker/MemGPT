from pgvector.psycopg import register_vector
from pgvector.sqlalchemy import Vector, JSON, Text
import psycopg


from sqlalchemy import create_engine, Column, String, BIGINT, select, inspect, text
from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import Column, BIGINT, String, DateTime

import re
from tqdm import tqdm
from typing import Optional, List, Iterator, Dict
import numpy as np
from tqdm import tqdm
import pandas as pd

from memgpt.config import MemGPTConfig
from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.config import AgentConfig, MemGPTConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.utils import printd
from memgpt.data_types import Record, Message, Passage

from datetime import datetime

Base = declarative_base()


def parse_formatted_time(formatted_time):
    # parse times returned by memgpt.utils.get_formatted_time()
    return datetime.strptime(formatted_time, "%Y-%m-%d %I:%M:%S %p %Z%z")


def get_db_model(table_name: str, table_type: TableType):
    config = MemGPTConfig.load()

    if table_name == TableType.ARCHIVAL_MEMORY:
        # create schema for archival memory
        class PassageModel(Base):
            """Defines data model for storing Passages (consisting of text, embedding)"""

            __abstract__ = True  # this line is necessary

            # Assuming passage_id is the primary key
            id = Column(BIGINT, primary_key=True, nullable=False, autoincrement=True)
            doc_id = Column(String)
            agent_id = Column(String)
            data_source = Column(String)  # agent_name if agent, data_source name if from data source
            text = Column(String, nullable=False)
            embedding = mapped_column(Vector(config.embedding_dim))
            metadata_ = Column(JSON(astext_type=Text()))

            def __repr__(self):
                return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model"
        Model = type(class_name, (PassageModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        return Model
    elif table_type == TableType.RECALL_MEMORY:

        class MessageModel(Base):
            """Defines data model for storing Message objects"""

            __abstract__ = True  # this line is necessary

            # Assuming message_id is the primary key
            id = Column(BIGINT, primary_key=True, nullable=False, autoincrement=True)
            user_id = Column(String, nullable=False)
            agent_id = Column(String, nullable=False)
            role = Column(String, nullable=False)
            content = Column(String, nullable=False)
            model = Column(String, nullable=False)
            function_name = Column(String)
            function_args = Column(String)
            function_response = Column(String)
            embedding = mapped_column(Vector(config.embedding_dim))

            # Add a datetime column, with default value as the current time
            created_at = Column(DateTime(timezone=True), server_default=func.now())

            def __repr__(self):
                return f"<Message(message_id='{self.id}', content='{self.content}', embedding='{self.embedding})>"

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model"
        Model = type(class_name, (MessageModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        return Model
    else:
        raise ValueError(f"Table type {table_type} not implemented")


class PostgresStorageConnector(StorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
        super().__init__(table_type=table_type, agent_config=agent_config)
        config = MemGPTConfig.load()

        # create table
        self.uri = config.archival_storage_uri
        if config.archival_storage_uri is None:
            raise ValueError(f"Must specifiy archival_storage_uri in config {config.config_path}")
        self.db_model = get_db_model(self.table_name)
        self.engine = create_engine(self.uri)
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)
        self.Session().execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def get_all_paginated(self, page_size: int, filters: Optional[Dict]) -> Iterator[List[Record]]:
        session = self.Session()
        offset = 0
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            db_passages_chunk = session.query(self.db_model).filter(*filters).offset(offset).limit(page_size).all()

            # If the chunk is empty, we've retrieved all records
            if not db_passages_chunk:
                break

            # Yield a list of Record objects converted from the chunk
            yield [self.type(**p.to_dict()) for p in db_passages_chunk]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self, limit=10, filters: Optional[Dict] = {}) -> List[Record]:
        session = self.Session()
        filters = self.get_filters(filters)
        db_passages = session.query(self.db_model).filter(*filters).limit(limit).all()
        return [self.type(**p.to_dict()) for p in db_passages]

    def get(self, id: str, filters: Optional[Dict] = {}) -> Optional[Passage]:
        session = self.Session()
        filters = self.get_filters(filters)
        db_passage = session.query(self.db_model).filter(*filters).get(id)
        if db_passage is None:
            return None
        return Passage(text=db_passage.text, embedding=db_passage.embedding, doc_id=db_passage.doc_id, passage_id=db_passage.passage_id)

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        session = self.Session()
        filters = self.get_filters(filters)
        return session.query(self.db_model).filter(*filters).count()

    def insert(self, passage: Passage):
        session = self.Session()
        db_passage = self.db_model(doc_id=passage.doc_id, text=passage.text, embedding=passage.embedding)
        session.add(db_passage)
        session.commit()

    def insert_many(self, records: List[Record], show_progress=True):
        session = self.Session()
        iterable = tqdm(records) if show_progress else records
        for passage in iterable:
            db_passage = self.db_model(doc_id=passage.doc_id, text=passage.text, embedding=passage.embedding)
            session.add(db_passage)
        session.commit()

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        session = self.Session()
        # Assuming PassageModel.embedding has the capability of computing l2_distance
        filters = self.get_filters(filters)
        results = session.scalars(
            select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)
        ).all()

        # Convert the results into Passage objects
        records = [self.type(**vars(result)) for result in results]
        return records

    def save(self):
        return

    def list_data_sources(self):
        assert self.table_type == TableType.ARCHIVAL_MEMORY, f"list_data_sources only implemented for ARCHIVAL_MEMORY"
        session = self.Session()
        unique_data_sources = session.query(self.db_model.data_source).filter(*self.filters).distinct().all()
        return unique_data_sources

    @staticmethod
    def list_loaded_data():
        config = MemGPTConfig.load()
        engine = create_engine(config.archival_storage_uri)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        tables = [table for table in tables if table.startswith("memgpt_") and not table.startswith("memgpt_agent_")]
        start_chars = len("memgpt_")
        tables = [table[start_chars:] for table in tables]
        return tables


class LanceDBConnector(StorageConnector):
    """Storage via LanceDB"""

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
        import lancedb

        self.db = lancedb.connect(self.uri)
        if self.table_name in self.db.table_names():
            self.table = self.db[self.table_name]
        else:
            self.table = None

    def get_all_paginated(self, page_size: int) -> Iterator[List[Passage]]:
        ds = self.table.to_lance()
        offset = 0
        while True:
            # Retrieve a chunk of records with the given page_size
            db_passages_chunk = ds.to_table(offset=offset, limit=page_size).to_pylist()
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
        db_passages = self.table.to_lance().to_table(limit=limit).to_pylist()
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
            return len(self.table)
        else:
            print(f"Table with name {self.table_name} not present")
            return 0

    def insert(self, passage: Passage):
        data = [{"doc_id": passage.doc_id, "text": passage.text, "passage_id": passage.passage_id, "vector": passage.embedding}]

        if self.table is not None:
            self.table.add(data)
        else:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")

    def insert_many(self, passages: List[Passage], show_progress=True):
        data = []
        iterable = tqdm(passages) if show_progress else passages
        for passage in iterable:
            temp_dict = {"doc_id": passage.doc_id, "text": passage.text, "passage_id": passage.passage_id, "vector": passage.embedding}
            data.append(temp_dict)

        if self.table is not None:
            self.table.add(data)
        else:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")

    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        # Assuming query_vec is of same length as embeddings inside table
        results = self.table.search(query_vec).limit(top_k).to_list()
        # Convert the results into Passage objects
        passages = [
            Passage(text=result["text"], embedding=result["vector"], doc_id=result["doc_id"], passage_id=result["passage_id"])
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
        start_chars = len("memgpt_")
        tables = [table[start_chars:] for table in tables]
        return tables
