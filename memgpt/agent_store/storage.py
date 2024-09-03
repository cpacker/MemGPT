""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""

import uuid
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Enum

from memgpt.config import MemGPTConfig

from memgpt.orm.base import Base as SQLBase
from memgpt.orm.message import Message as SQLMessage
from memgpt.orm.passage import Passage as SQLPassage
from memgpt.orm.document import Document as SQLDocument
from memgpt.orm.utilities import get_db_session

from memgpt.schemas.enums import TableType

if TYPE_CHECKING:
    from sqlalchemy.orm import Session



class StorageConnector:
    """Defines a DB connection that is user-specific to access data: Documents, Passages, Archival/Recall Memory"""

    SQLModel: SQLBase
    db_session: "Session" = None

    def __init__(
        self,
        table_type: TableType,
        config: MemGPTConfig,
        user_id: str,
        agent_id: str = None,
        db_session: Optional["Session"] = None
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.table_type = table_type

        self.db_session = db_session or get_db_session()

        match table_type:
            case TableType.ARCHIVAL_MEMORY:
                self.SQLModel = SQLPassage
            case TableType.RECALL_MEMORY:
                self.SQLModel = SQLMessage
            case TableType.DOCUMENTS:
                self.SQLModel = SQLDocument
            case TableType.PASSAGES:
                self.SQLModel = SQLPassage
            case _:
                raise ValueError(f"Table type {table_type} not implemented")

        # setup base filters for agent-specific tables
        if self.table_type == TableType.ARCHIVAL_MEMORY or self.table_type == TableType.RECALL_MEMORY:
            # agent-specific table
            assert agent_id is not None, "Agent ID must be provided for agent-specific tables"
            self.filters = {"user_id": self.user_id, "agent_id": self.agent_id}
        elif self.table_type == TableType.PASSAGES or self.table_type == TableType.DOCUMENTS:
            # setup base filters for user-specific tables
            assert agent_id is None, "Agent ID must not be provided for user-specific tables"
            self.filters = {"user_id": self.user_id}
        else:
            raise ValueError(f"Table type {table_type} not implemented")

    @staticmethod
    def get_storage_connector(
        table_type: TableType,
        config: MemGPTConfig,
        user_id: str,
        agent_id: str = None,
    ):
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            storage_type = config.archival_storage_type
        elif table_type == TableType.RECALL_MEMORY:
            storage_type = config.recall_storage_type
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        match storage_type:
            case "postgres":
                from memgpt.agent_store.db import PostgresStorageConnector
                return PostgresStorageConnector(table_type, config, user_id, agent_id)
            case "sqlite":
                from memgpt.agent_store.db import SQLLiteStorageConnector
                return SQLLiteStorageConnector(table_type, config, user_id, agent_id)
            # TODO: implement other storage types
            # case "chroma":
            #     from memgpt.agent_store.chroma import ChromaStorageConnector
            #     return ChromaStorageConnector(table_type, config, user_id, agent_id)
            case _:
                raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def get_archival_storage_connector(user_id, agent_id):
        config = MemGPTConfig.load()
        return StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config, user_id, agent_id)

    @staticmethod
    def get_recall_storage_connector(user_id, agent_id):
        config = MemGPTConfig.load()
        return StorageConnector.get_storage_connector(TableType.RECALL_MEMORY, config, user_id, agent_id)

    @abstractmethod
    def get_filters(self, filters: Optional[Dict] = {}) -> Union[Tuple[list, dict], dict]:
        pass

    @abstractmethod
    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: int = 1000):
        pass

    @abstractmethod
    def get_all(self, filters: Optional[Dict] = {}, limit=10):
        pass

    @abstractmethod
    def get(self, id: uuid.UUID):
        pass

    @abstractmethod
    def size(self, filters: Optional[Dict] = {}) -> int:
        pass

    @abstractmethod
    def insert(self, record):
        pass

    @abstractmethod
    def insert_many(self, records, show_progress=False):
        pass

    @abstractmethod
    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}):
        pass

    @abstractmethod
    def query_date(self, start_date, end_date):
        pass

    @abstractmethod
    def query_text(self, query):
        pass

    @abstractmethod
    def delete_table(self):
        pass

    @abstractmethod
    def delete(self, filters: Optional[Dict] = {}):
        pass

    @abstractmethod
    def save(self):
        pass
