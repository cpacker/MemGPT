import base64
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import (
    BINARY,
    Column,
    DateTime,
    Index,
    String,
    TypeDecorator,
    and_,
    asc,
    desc,
    or_,
    select,
    text,
)
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm.session import close_all_sessions
from sqlalchemy.sql import func
from sqlalchemy_json import MutableJson
from tqdm import tqdm

from letta.agent_store.storage import StorageConnector, TableType
from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.metadata import EmbeddingConfigColumn, ToolCallColumn
from letta.orm.base import Base
from letta.orm.file import FileMetadata as FileMetadataModel

# from letta.schemas.message import Message, Passage, Record, RecordType, ToolCall
from letta.schemas.message import Message
from letta.schemas.openai.chat_completions import ToolCall
from letta.schemas.passage import Passage
from letta.settings import settings

config = LettaConfig()


class CommonVector(TypeDecorator):
    """Common type for representing vectors in SQLite"""

    impl = BINARY
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(BINARY())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        # Ensure value is a numpy array
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        # Serialize numpy array to bytes, then encode to base64 for universal compatibility
        return base64.b64encode(value.tobytes())

    def process_result_value(self, value, dialect):
        if not value:
            return value
        # Check database type and deserialize accordingly
        if dialect.name == "sqlite":
            # Decode from base64 and convert back to numpy array
            value = base64.b64decode(value)
        # For PostgreSQL, value is already in bytes
        return np.frombuffer(value, dtype=np.float32)


class MessageModel(Base):
    """Defines data model for storing Message objects"""

    __tablename__ = "messages"
    __table_args__ = {"extend_existing": True}

    # Assuming message_id is the primary key
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)

    # openai info
    role = Column(String, nullable=False)
    text = Column(String)  # optional: can be null if function call
    model = Column(String)  # optional: can be null if LLM backend doesn't require specifying
    name = Column(String)  # optional: multi-agent only

    # tool call request info
    # if role == "assistant", this MAY be specified
    # if role != "assistant", this must be null
    # TODO align with OpenAI spec of multiple tool calls
    # tool_calls = Column(ToolCallColumn)
    tool_calls = Column(ToolCallColumn)

    # tool call response info
    # if role == "tool", then this must be specified
    # if role != "tool", this must be null
    tool_call_id = Column(String)

    # Add a datetime column, with default value as the current time
    created_at = Column(DateTime(timezone=True))
    Index("message_idx_user", user_id, agent_id),

    def __repr__(self):
        return f"<Message(message_id='{self.id}', text='{self.text}')>"

    def to_record(self):
        # calls = (
        #    [ToolCall(id=tool_call["id"], function=ToolCallFunction(**tool_call["function"])) for tool_call in self.tool_calls]
        #    if self.tool_calls
        #    else None
        # )
        # if calls:
        #    assert isinstance(calls[0], ToolCall)
        if self.tool_calls and len(self.tool_calls) > 0:
            assert isinstance(self.tool_calls[0], ToolCall), type(self.tool_calls[0])
            for tool in self.tool_calls:
                assert isinstance(tool, ToolCall), type(tool)
        return Message(
            user_id=self.user_id,
            agent_id=self.agent_id,
            role=self.role,
            name=self.name,
            text=self.text,
            model=self.model,
            # tool_calls=[ToolCall(id=tool_call["id"], function=ToolCallFunction(**tool_call["function"])) for tool_call in self.tool_calls] if self.tool_calls else None,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            created_at=self.created_at,
            id=self.id,
        )


class PassageModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "passages"
    __table_args__ = {"extend_existing": True}

    # Assuming passage_id is the primary key
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    text = Column(String)
    file_id = Column(String)
    agent_id = Column(String)
    source_id = Column(String)

    # vector storage
    if settings.letta_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
    elif config.archival_storage_type == "sqlite" or config.archival_storage_type == "chroma":
        embedding = Column(CommonVector)
    else:
        raise ValueError(f"Unsupported archival_storage_type: {config.archival_storage_type}")
    embedding_config = Column(EmbeddingConfigColumn)
    metadata_ = Column(MutableJson)

    # Add a datetime column, with default value as the current time
    created_at = Column(DateTime(timezone=True))

    Index("passage_idx_user", user_id, agent_id, file_id),

    def __repr__(self):
        return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

    def to_record(self):
        return Passage(
            text=self.text,
            embedding=self.embedding,
            embedding_config=self.embedding_config,
            file_id=self.file_id,
            user_id=self.user_id,
            id=self.id,
            source_id=self.source_id,
            agent_id=self.agent_id,
            metadata_=self.metadata_,
            created_at=self.created_at,
        )


class SQLStorageConnector(StorageConnector):
    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        self.config = config

    def get_filters(self, filters: Optional[Dict] = {}):
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        all_filters = [getattr(self.db_model, key) == value for key, value in filter_conditions.items()]
        return all_filters

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000, offset=0):
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            with self.session_maker() as session:
                db_record_chunk = session.query(self.db_model).filter(*filters).offset(offset).limit(page_size).all()

            # If the chunk is empty, we've retrieved all records
            if not db_record_chunk:
                break

            # Yield a list of Record objects converted from the chunk
            yield [record.to_record() for record in db_record_chunk]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all_cursor(
        self,
        filters: Optional[Dict] = {},
        after: str = None,
        before: str = None,
        limit: Optional[int] = 1000,
        order_by: str = "created_at",
        reverse: bool = False,
    ):
        """Get all that returns a cursor (record.id) and records"""
        filters = self.get_filters(filters)

        # generate query
        with self.session_maker() as session:
            query = session.query(self.db_model).filter(*filters)
            # query = query.order_by(asc(self.db_model.id))

            # records are sorted by the order_by field first, and then by the ID if two fields are the same
            if reverse:
                query = query.order_by(desc(getattr(self.db_model, order_by)), asc(self.db_model.id))
            else:
                query = query.order_by(asc(getattr(self.db_model, order_by)), asc(self.db_model.id))

            # cursor logic: filter records based on before/after ID
            if after:
                after_value = getattr(self.get(id=after), order_by)
                sort_exp = getattr(self.db_model, order_by) > after_value
                query = query.filter(
                    or_(sort_exp, and_(getattr(self.db_model, order_by) == after_value, self.db_model.id > after))  # tiebreaker case
                )
            if before:
                before_value = getattr(self.get(id=before), order_by)
                sort_exp = getattr(self.db_model, order_by) < before_value
                query = query.filter(or_(sort_exp, and_(getattr(self.db_model, order_by) == before_value, self.db_model.id < before)))

            # get records
            db_record_chunk = query.limit(limit).all()
        if not db_record_chunk:
            return (None, [])
        records = [record.to_record() for record in db_record_chunk]
        next_cursor = db_record_chunk[-1].id
        assert isinstance(next_cursor, str)

        # return (cursor, list[records])
        return (next_cursor, records)

    def get_all(self, filters: Optional[Dict] = {}, limit=None):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            if limit:
                db_records = session.query(self.db_model).filter(*filters).limit(limit).all()
            else:
                db_records = session.query(self.db_model).filter(*filters).all()
        return [record.to_record() for record in db_records]

    def get(self, id: str):
        with self.session_maker() as session:
            db_record = session.get(self.db_model, id)
        if db_record is None:
            return None
        return db_record.to_record()

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            return session.query(self.db_model).filter(*filters).count()

    def insert(self, record):
        raise NotImplementedError

    def insert_many(self, records, show_progress=False):
        raise NotImplementedError

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}):
        raise NotImplementedError("Vector query not implemented for SQLStorageConnector")

    def save(self):
        return

    def list_data_sources(self):
        assert self.table_type == TableType.ARCHIVAL_MEMORY, f"list_data_sources only implemented for ARCHIVAL_MEMORY"
        with self.session_maker() as session:
            unique_data_sources = session.query(self.db_model.data_source).filter(*self.filters).distinct().all()
        return unique_data_sources

    def query_date(self, start_date, end_date, limit=None, offset=0):
        filters = self.get_filters({})
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(self.db_model.created_at >= start_date)
                .filter(self.db_model.created_at <= end_date)
                .filter(self.db_model.role != "system")
                .filter(self.db_model.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return [result.to_record() for result in results]

    def query_text(self, query, limit=None, offset=0):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        filters = self.get_filters({})
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(func.lower(self.db_model.text).contains(func.lower(query)))
                .filter(self.db_model.role != "system")
                .filter(self.db_model.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        # return [self.type(**vars(result)) for result in results]
        return [result.to_record() for result in results]

    # Should be used only in tests!
    def delete_table(self):
        close_all_sessions()
        with self.session_maker() as session:
            self.db_model.__table__.drop(session.bind)
            session.commit()

    def delete(self, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            session.query(self.db_model).filter(*filters).delete()
            session.commit()


class PostgresStorageConnector(SQLStorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        from pgvector.sqlalchemy import Vector

        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        # construct URI from enviornment variables
        if settings.pg_uri:
            self.uri = settings.pg_uri

        # use config URI
        # TODO: remove this eventually (config should NOT contain URI)
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            self.uri = self.config.archival_storage_uri
            self.db_model = PassageModel
            if self.config.archival_storage_uri is None:
                raise ValueError(f"Must specify archival_storage_uri in config {self.config.config_path}")
        elif table_type == TableType.RECALL_MEMORY:
            self.uri = self.config.recall_storage_uri
            self.db_model = MessageModel
            if self.config.recall_storage_uri is None:
                raise ValueError(f"Must specify recall_storage_uri in config {self.config.config_path}")
        elif table_type == TableType.FILES:
            self.uri = self.config.metadata_storage_uri
            self.db_model = FileMetadataModel
            if self.config.metadata_storage_uri is None:
                raise ValueError(f"Must specify metadata_storage_uri in config {self.config.config_path}")
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        for c in self.db_model.__table__.columns:
            if c.name == "embedding":
                assert isinstance(c.type, Vector), f"Embedding column must be of type Vector, got {c.type}"

        from letta.server.server import db_context

        self.session_maker = db_context

        # TODO: move to DB init
        with self.session_maker() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            results = session.scalars(
                select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)
            ).all()

        # Convert the results into Passage objects
        records = [result.to_record() for result in results]
        return records

    def insert_many(self, records, exists_ok=True, show_progress=False):
        # TODO: this is terrible, should eventually be done the same way for all types (migrate to SQLModel)
        if len(records) == 0:
            return

        added_ids = []  # avoid adding duplicates
        # NOTE: this has not great performance due to the excessive commits
        with self.session_maker() as session:
            iterable = tqdm(records) if show_progress else records
            for record in iterable:
                # db_record = self.db_model(**vars(record))

                if record.id in added_ids:
                    continue

                existing_record = session.query(self.db_model).filter_by(id=record.id).first()
                if existing_record:
                    if exists_ok:
                        fields = record.model_dump()
                        fields.pop("id")
                        session.query(self.db_model).filter(self.db_model.id == record.id).update(fields)
                        print(f"Updated record with id {record.id}")
                        session.commit()
                    else:
                        raise ValueError(f"Record with id {record.id} already exists.")

                else:
                    db_record = self.db_model(**record.dict())
                    session.add(db_record)
                    print(f"Added record with id {record.id}")
                    session.commit()

                added_ids.append(record.id)

    def insert(self, record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def update(self, record):
        """
        Updates a record in the database based on the provided Record object.
        """
        with self.session_maker() as session:
            # Find the record by its ID
            db_record = session.query(self.db_model).filter_by(id=record.id).first()
            if not db_record:
                raise ValueError(f"Record with id {record.id} does not exist.")

            # Update the record with new values from the provided Record object
            for attr, value in vars(record).items():
                setattr(db_record, attr, value)

            # Commit the changes to the database
            session.commit()

    def str_to_datetime(self, str_date: str) -> datetime:
        val = str_date.split("-")
        _datetime = datetime(int(val[0]), int(val[1]), int(val[2]))
        return _datetime

    def query_date(self, start_date, end_date, limit=None, offset=0):
        filters = self.get_filters({})
        _start_date = self.str_to_datetime(start_date) if isinstance(start_date, str) else start_date
        _end_date = self.str_to_datetime(end_date) if isinstance(end_date, str) else end_date
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(self.db_model.created_at >= _start_date)
                .filter(self.db_model.created_at <= _end_date)
                .filter(self.db_model.role != "system")
                .filter(self.db_model.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return [result.to_record() for result in results]


class SQLLiteStorageConnector(SQLStorageConnector):
    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        # get storage URI
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            raise ValueError(f"Table type {table_type} not implemented")
        elif table_type == TableType.RECALL_MEMORY:
            # TODO: eventually implement URI option
            self.path = self.config.recall_storage_path
            if self.path is None:
                raise ValueError(f"Must specify recall_storage_path in config.")
            self.db_model = MessageModel
        elif table_type == TableType.FILES:
            self.path = self.config.metadata_storage_path
            if self.path is None:
                raise ValueError(f"Must specify metadata_storage_path in config.")
            self.db_model = FileMetadataModel

        else:
            raise ValueError(f"Table type {table_type} not implemented")

        self.path = os.path.join(self.path, f"sqlite.db")

        from letta.server.server import db_context

        self.session_maker = db_context

        # Need this in order to allow UUIDs to be stored successfully in the sqlite database
        # import sqlite3
        # import uuid
        #
        # sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)
        # sqlite3.register_converter("UUID", lambda b: uuid.UUID(bytes_le=b))

    def insert_many(self, records, exists_ok=True, show_progress=False):
        # TODO: this is terrible, should eventually be done the same way for all types (migrate to SQLModel)
        if len(records) == 0:
            return

        added_ids = []  # avoid adding duplicates
        # NOTE: this has not great performance due to the excessive commits
        with self.session_maker() as session:
            iterable = tqdm(records) if show_progress else records
            for record in iterable:
                # db_record = self.db_model(**vars(record))

                if record.id in added_ids:
                    continue

                existing_record = session.query(self.db_model).filter_by(id=record.id).first()
                if existing_record:
                    if exists_ok:
                        fields = record.model_dump()
                        fields.pop("id")
                        session.query(self.db_model).filter(self.db_model.id == record.id).update(fields)
                        session.commit()
                    else:
                        raise ValueError(f"Record with id {record.id} already exists.")

                else:
                    db_record = self.db_model(**record.dict())
                    session.add(db_record)
                    session.commit()

                added_ids.append(record.id)

    def insert(self, record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def update(self, record):
        """
        Updates an existing record in the database with values from the provided record object.
        """
        if not record.id:
            raise ValueError("Record must have an id.")

        with self.session_maker() as session:
            # Fetch the existing record from the database
            db_record = session.query(self.db_model).filter_by(id=record.id).first()
            if not db_record:
                raise ValueError(f"Record with id {record.id} does not exist.")

            # Update the database record with values from the provided record object
            for column in self.db_model.__table__.columns:
                column_name = column.name
                if hasattr(record, column_name):
                    new_value = getattr(record, column_name)
                    setattr(db_record, column_name, new_value)

            # Commit the changes to the database
            session.commit()


def attach_base():
    # This should be invoked in server.py to make sure Base gets initialized properly
    # DO NOT REMOVE
    from letta.utils import printd

    printd("Initializing database...")
