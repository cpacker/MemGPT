from pgvector.psycopg import register_vector
import os
from pgvector.sqlalchemy import Vector
import psycopg


from sqlalchemy import create_engine, Column, String, BIGINT, select, inspect, text, JSON
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import Column, BIGINT, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy_json import mutable_json_type
from sqlalchemy import TypeDecorator, CHAR
import uuid

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
from memgpt.data_types import Record, Message, Passage, Source

from datetime import datetime


# Custom UUID type
class CommonUUID(TypeDecorator):
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR())

    def process_bind_param(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return str(value)  # Convert UUID to string for SQLite

    def process_result_value(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return uuid.UUID(value)


Base = declarative_base()


def get_db_model(table_name: str, table_type: TableType):
    config = MemGPTConfig.load()

    if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
        # create schema for archival memory
        class PassageModel(Base):
            """Defines data model for storing Passages (consisting of text, embedding)"""

            __abstract__ = True  # this line is necessary

            # Assuming passage_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(String, nullable=False)
            text = Column(String, nullable=False)
            doc_id = Column(String)
            agent_id = Column(String)
            data_source = Column(String)  # agent_name if agent, data_source name if from data source
            embedding = mapped_column(Vector(config.embedding_dim))
            metadata_ = Column(mutable_json_type(dbtype=JSONB, nested=True))

            def __repr__(self):
                return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

            def to_record(self):
                return Passage(
                    text=self.text,
                    embedding=self.embedding,
                    doc_id=self.doc_id,
                    user_id=self.user_id,
                    id=self.id,
                    data_source=self.data_source,
                    agent_id=self.agent_id,
                    metadata=self.metadata_,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model"
        Model = type(class_name, (PassageModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        return Model
    elif table_type == TableType.RECALL_MEMORY:

        class MessageModel(Base):
            """Defines data model for storing Message objects"""

            __abstract__ = True  # this line is necessary

            # Assuming message_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(String, nullable=False)
            agent_id = Column(String, nullable=False)

            # openai info
            role = Column(String, nullable=False)
            text = Column(String, nullable=False)
            model = Column(String, nullable=False)
            user = Column(String)  # optional: multi-agent only

            # function info
            function_name = Column(String)
            function_args = Column(String)
            function_response = Column(String)

            embedding = mapped_column(Vector(config.embedding_dim))

            # Add a datetime column, with default value as the current time
            created_at = Column(DateTime(timezone=True), server_default=func.now())

            def __repr__(self):
                return f"<Message(message_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

            def to_record(self):
                return Message(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    user=self.user,
                    text=self.text,
                    model=self.model,
                    function_name=self.function_name,
                    function_args=self.function_args,
                    function_response=self.function_response,
                    embedding=self.embedding,
                    created_at=self.created_at,
                    id=self.id,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model"
        Model = type(class_name, (MessageModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        return Model
    elif table_type == TableType.DATA_SOURCES:

        class SourceModel(Base):
            """Defines data model for storing Passages (consisting of text, embedding)"""

            __abstract__ = True  # this line is necessary

            # Assuming passage_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            user_id = Column(String, nullable=False)
            name = Column(String, nullable=False)
            created_at = Column(DateTime(timezone=True), server_default=func.now())

            def __repr__(self):
                return f"<Source(passage_id='{self.id}', name='{self.name}')>"

            def to_record(self):
                return Source(id=self.id, user_id=self.user_id, name=self.name, created_at=self.created_at)

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model"
        Model = type(class_name, (SourceModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        return Model
    else:
        raise ValueError(f"Table type {table_type} not implemented")


class SQLStorageConnector(StorageConnector):
    def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
        super().__init__(table_type=table_type, agent_config=agent_config)
        self.config = MemGPTConfig.load()

    def get_filters(self, filters: Optional[Dict] = {}):
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        return [getattr(self.db_model, key) == value for key, value in filter_conditions.items()]

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000) -> Iterator[List[Record]]:
        session = self.Session()
        offset = 0
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            db_record_chunk = session.query(self.db_model).filter(*filters).offset(offset).limit(page_size).all()

            # If the chunk is empty, we've retrieved all records
            if not db_record_chunk:
                break

            # Yield a list of Record objects converted from the chunk
            yield [record.to_record() for record in db_record_chunk]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self, filters: Optional[Dict] = {}, limit=None) -> List[Record]:
        session = self.Session()
        filters = self.get_filters(filters)
        if limit:
            db_records = session.query(self.db_model).filter(*filters).limit(limit).all()
        else:
            db_records = session.query(self.db_model).filter(*filters).all()
        return [record.to_record() for record in db_records]

    def get(self, id: str) -> Optional[Record]:
        session = self.Session()
        db_record = session.query(self.db_model).get(id)
        if db_record is None:
            return None
        return db_record.to_record()

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        session = self.Session()
        filters = self.get_filters(filters)
        return session.query(self.db_model).filter(*filters).count()

    def insert(self, record: Record):
        session = self.Session()
        db_record = self.db_model(**vars(record))
        session.add(db_record)
        session.commit()

    def insert_many(self, records: List[Record], show_progress=True):
        session = self.Session()
        iterable = tqdm(records) if show_progress else records
        for record in iterable:
            db_record = self.db_model(**vars(record))
            session.add(db_record)
        session.commit()

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        raise NotImplementedError("Vector query not implemented for SQLStorageConnector")

    def save(self):
        return

    def list_data_sources(self):
        assert self.table_type == TableType.ARCHIVAL_MEMORY, f"list_data_sources only implemented for ARCHIVAL_MEMORY"
        session = self.Session()
        unique_data_sources = session.query(self.db_model.data_source).filter(*self.filters).distinct().all()
        return unique_data_sources

    def query_date(self, start_date, end_date):
        session = self.Session()
        filters = self.get_filters({})
        results = (
            session.query(self.db_model)
            .filter(*filters)
            .filter(self.db_model.created_at >= start_date)
            .filter(self.db_model.created_at <= end_date)
            .all()
        )
        return [result.to_record() for result in results]

    def query_text(self, query):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        session = self.Session()
        filters = self.get_filters({})
        results = session.query(self.db_model).filter(*filters).filter(func.lower(self.db_model.text).contains(func.lower(query))).all()
        # return [self.type(**vars(result)) for result in results]
        return [result.to_record() for result in results]

    def delete_table(self):
        session = self.Session()
        self.db_model.__table__.drop(session.bind)
        session.commit()

    def delete(self, filters: Optional[Dict] = {}):
        session = self.Session()
        filters = self.get_filters(filters)
        session.query(self.db_model).filter(*filters).delete()
        session.commit()


class PostgresStorageConnector(SQLStorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
        super().__init__(table_type=table_type, agent_config=agent_config)

        # get storage URI
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            self.uri = self.config.archival_storage_uri
            if self.config.archival_storage_uri is None:
                raise ValueError(f"Must specifiy archival_storage_uri in config {self.config.config_path}")
        elif table_type == TableType.RECALL_MEMORY:
            self.uri = self.config.recall_storage_uri
            if self.config.recall_storage_uri is None:
                raise ValueError(f"Must specifiy recall_storage_uri in config {self.config.config_path}")
        elif table_type == TableType.DATA_SOURCES:
            self.uri = self.config.metadata_storage_uri
            if self.config.metadata_storage_uri is None:
                raise ValueError(f"Must specifiy metadata_storage_uri in config {self.config.config_path}")
        else:
            raise ValueError(f"Table type {table_type} not implemented")
        # create table
        self.db_model = get_db_model(self.table_name, table_type)
        self.engine = create_engine(self.uri)
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)
        self.Session().execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        session = self.Session()
        filters = self.get_filters(filters)
        results = session.scalars(
            select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)
        ).all()

        # Convert the results into Passage objects
        records = [result.to_record() for result in results]
        return records


class SQLLiteStorageConnector(SQLStorageConnector):
    def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
        super().__init__(table_type=table_type, agent_config=agent_config)

        # get storage URI
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            raise ValueError(f"Table type {table_type} not implemented")
        elif table_type == TableType.RECALL_MEMORY:
            # TODO: eventually implement URI option
            self.path = self.config.recall_storage_path
            if self.path is None:
                raise ValueError(f"Must specifiy recall_storage_path in config {self.config.recall_storage_path}")
        elif table_type == TableType.DATA_SOURCES:
            self.path = self.config.metadata_storage_path
            if self.path is None:
                raise ValueError(f"Must specifiy metadata_storage_path in config {self.config.metadata_storage_path}")
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        self.path = os.path.join(self.path, f"{self.table_name}.db")
        self.db_model = get_db_model(self.table_name, table_type)

        # Create the SQLAlchemy engine
        self.db_model = get_db_model(self.table_name, table_type)
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)

        import sqlite3

        sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)
        sqlite3.register_converter("UUID", lambda b: uuid.UUID(bytes_le=b))
