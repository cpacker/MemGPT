import os
import ast
import psycopg


from sqlalchemy import create_engine, Column, String, BIGINT, select, inspect, text, JSON, BLOB, BINARY, ARRAY, DateTime
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, mapped_column, declarative_base
from sqlalchemy.orm.session import close_all_sessions
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy_json import mutable_json_type, MutableJson
from sqlalchemy import TypeDecorator, CHAR
import uuid

import re
from tqdm import tqdm
from typing import Optional, List, Iterator, Dict
import numpy as np
from tqdm import tqdm
import pandas as pd

from memgpt.config import MemGPTConfig
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.config import MemGPTConfig
from memgpt.utils import printd
from memgpt.data_types import Record, Message, Passage, ToolCall
from memgpt.metadata import MetadataStore

from datetime import datetime


# Custom UUID type
class CommonUUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

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


class CommonVector(TypeDecorator):
    """Common type for representing vectors in SQLite"""

    impl = BINARY
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(BINARY())

    def process_bind_param(self, value, dialect):
        if value:
            assert isinstance(value, np.ndarray) or isinstance(value, list), f"Value must be of type np.ndarray or list, got {type(value)}"
            assert isinstance(value[0], float), f"Value must be of type float, got {type(value[0])}"
            # print("WRITE", np.array(value).tobytes())
            return np.array(value).tobytes()
        else:
            # print("WRITE", value, type(value))
            return value

    def process_result_value(self, value, dialect):
        if not value:
            return value
        # print("dialect", dialect, type(value))
        return np.frombuffer(value)


# Custom serialization / de-serialization for JSON columns


class ToolCallColumn(TypeDecorator):
    """Custom type for storing List[ToolCall] as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return [vars(v) for v in value]
        return value

    def process_result_value(self, value, dialect):
        if value:
            return [ToolCall(**v) for v in value]
        return value


Base = declarative_base()


def get_db_model(config: MemGPTConfig, table_name: str, table_type: TableType, user_id, agent_id=None, dialect="postgresql"):
    # get embedding dimention info
    ms = MetadataStore(config)
    if agent_id and ms.get_agent(agent_id):
        agent = ms.get_agent(agent_id)
        embedding_dim = agent.embedding_config.embedding_dim
    else:
        user = ms.get_user(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        embedding_dim = user.default_embedding_config.embedding_dim

    # Define a helper function to create or get the model class
    def create_or_get_model(class_name, base_model, table_name):
        if class_name in globals():
            return globals()[class_name]
        Model = type(class_name, (base_model,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        globals()[class_name] = Model
        return Model

    if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
        # create schema for archival memory
        class PassageModel(Base):
            """Defines data model for storing Passages (consisting of text, embedding)"""

            __abstract__ = True  # this line is necessary

            # Assuming passage_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(CommonUUID, nullable=False)
            text = Column(String, nullable=False)
            doc_id = Column(CommonUUID)
            agent_id = Column(CommonUUID)
            data_source = Column(String)  # agent_name if agent, data_source name if from data source

            # vector storage
            if dialect == "sqlite":
                embedding = Column(CommonVector)
            else:
                from pgvector.sqlalchemy import Vector

                embedding = mapped_column(Vector(embedding_dim))

            metadata_ = Column(MutableJson)

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
        class_name = f"{table_name.capitalize()}Model" + dialect
        return create_or_get_model(class_name, PassageModel, table_name)

    elif table_type == TableType.RECALL_MEMORY:

        class MessageModel(Base):
            """Defines data model for storing Message objects"""

            __abstract__ = True  # this line is necessary

            # Assuming message_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(CommonUUID, nullable=False)
            agent_id = Column(CommonUUID, nullable=False)

            # openai info
            role = Column(String, nullable=False)
            text = Column(String)  # optional: can be null if function call
            model = Column(String, nullable=False)
            name = Column(String)  # optional: multi-agent only

            # tool call request info
            # if role == "assistant", this MAY be specified
            # if role != "assistant", this must be null
            # TODO align with OpenAI spec of multiple tool calls
            tool_calls = Column(ToolCallColumn)

            # tool call response info
            # if role == "tool", then this must be specified
            # if role != "tool", this must be null
            tool_call_id = Column(String)

            # vector storage
            if dialect == "sqlite":
                embedding = Column(CommonVector)
            else:
                from pgvector.sqlalchemy import Vector

                embedding = mapped_column(Vector(embedding_dim))

            # Add a datetime column, with default value as the current time
            created_at = Column(DateTime(timezone=True), server_default=func.now())

            def __repr__(self):
                return f"<Message(message_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

            def to_record(self):
                return Message(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    name=self.name,
                    text=self.text,
                    model=self.model,
                    tool_calls=self.tool_calls,
                    tool_call_id=self.tool_call_id,
                    embedding=self.embedding,
                    created_at=self.created_at,
                    id=self.id,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model" + dialect
        return create_or_get_model(class_name, MessageModel, table_name)

    else:
        raise ValueError(f"Table type {table_type} not implemented")


class SQLStorageConnector(StorageConnector):
    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        self.config = config

    def get_filters(self, filters: Optional[Dict] = {}):
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        all_filters = [getattr(self.db_model, key) == value for key, value in filter_conditions.items()]
        return all_filters

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000) -> Iterator[List[Record]]:
        offset = 0
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            db_record_chunk = self.session.query(self.db_model).filter(*filters).offset(offset).limit(page_size).all()

            # If the chunk is empty, we've retrieved all records
            if not db_record_chunk:
                break

            # Yield a list of Record objects converted from the chunk
            yield [record.to_record() for record in db_record_chunk]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self, filters: Optional[Dict] = {}, limit=None) -> List[Record]:
        filters = self.get_filters(filters)
        if limit:
            db_records = self.session.query(self.db_model).filter(*filters).limit(limit).all()
        else:
            db_records = self.session.query(self.db_model).filter(*filters).all()
        return [record.to_record() for record in db_records]

    def get(self, id: str) -> Optional[Record]:
        db_record = self.session.query(self.db_model).get(id)
        if db_record is None:
            return None
        return db_record.to_record()

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        filters = self.get_filters(filters)
        return self.session.query(self.db_model).filter(*filters).count()

    def insert(self, record: Record):
        db_record = self.db_model(**vars(record))
        self.session.add(db_record)
        self.session.commit()

    def insert_many(self, records: List[Record], show_progress=False):
        iterable = tqdm(records) if show_progress else records
        for record in iterable:
            db_record = self.db_model(**vars(record))
            self.session.add(db_record)
        self.session.commit()

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        raise NotImplementedError("Vector query not implemented for SQLStorageConnector")

    def save(self):
        return

    def list_data_sources(self):
        assert self.table_type == TableType.ARCHIVAL_MEMORY, f"list_data_sources only implemented for ARCHIVAL_MEMORY"
        unique_data_sources = self.session.query(self.db_model.data_source).filter(*self.filters).distinct().all()
        return unique_data_sources

    def query_date(self, start_date, end_date, offset=0, limit=None):
        filters = self.get_filters({})
        query = (
            self.session.query(self.db_model)
            .filter(*filters)
            .filter(self.db_model.created_at >= start_date)
            .filter(self.db_model.created_at <= end_date)
            .offset(offset)
        )
        if limit:
            query = query.limit(limit)
        results = query.all()
        return [result.to_record() for result in results]

    def query_text(self, query, offset=0, limit=None):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        filters = self.get_filters({})
        query = (
            self.session.query(self.db_model)
            .filter(*filters)
            .filter(func.lower(self.db_model.text).contains(func.lower(query)))
            .offset(offset)
        )
        if limit:
            query = query.limit(limit)
        results = query.all()
        # return [self.type(**vars(result)) for result in results]
        return [result.to_record() for result in results]

    def delete_table(self):
        close_all_sessions()
        self.db_model.__table__.drop(self.session.bind)
        self.session.commit()

    def delete(self, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        self.session.query(self.db_model).filter(*filters).delete()
        self.session.commit()


class PostgresStorageConnector(SQLStorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        from pgvector.sqlalchemy import Vector

        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

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
        self.db_model = get_db_model(config, self.table_name, table_type, user_id, agent_id)
        self.engine = create_engine(self.uri)
        for c in self.db_model.__table__.columns:
            if c.name == "embedding":
                assert isinstance(c.type, Vector), f"Embedding column must be of type Vector, got {c.type}"
        Base.metadata.create_all(self.engine, tables=[self.db_model.__table__])  # Create the table if it doesn't exist

        session_maker = sessionmaker(bind=self.engine)
        self.session = session_maker()
        self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        filters = self.get_filters(filters)
        results = self.session.scalars(
            select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)
        ).all()

        # Convert the results into Passage objects
        records = [result.to_record() for result in results]
        return records


class SQLLiteStorageConnector(SQLStorageConnector):
    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        # get storage URI
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            raise ValueError(f"Table type {table_type} not implemented")
        elif table_type == TableType.RECALL_MEMORY:
            # TODO: eventually implement URI option
            self.path = self.config.recall_storage_path
            if self.path is None:
                raise ValueError(f"Must specifiy recall_storage_path in config {self.config.recall_storage_path}")
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        self.path = os.path.join(self.path, f"{self.table_name}.db")

        # Create the SQLAlchemy engine
        self.db_model = get_db_model(config, self.table_name, table_type, user_id, agent_id, dialect="sqlite")
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(self.engine, tables=[self.db_model.__table__])  # Create the table if it doesn't exist
        session_maker = sessionmaker(bind=self.engine)
        self.session = session_maker()

        import sqlite3

        sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)
        sqlite3.register_converter("UUID", lambda b: uuid.UUID(bytes_le=b))
