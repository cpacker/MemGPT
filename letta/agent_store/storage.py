""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""

import uuid
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from letta.config import LettaConfig
from letta.schemas.file import FileMetadata
from letta.schemas.message import Message
from letta.schemas.passage import Passage
from letta.utils import printd


# ENUM representing table types in Letta
# each table corresponds to a different table schema  (specified in data_types.py)
class TableType:
    ARCHIVAL_MEMORY = "archival_memory"  # recall memory table: letta_agent_{agent_id}
    RECALL_MEMORY = "recall_memory"  # archival memory table: letta_agent_recall_{agent_id}
    PASSAGES = "passages"  # TODO
    FILES = "files"


# table names used by Letta

# agent tables
RECALL_TABLE_NAME = "letta_recall_memory_agent"  # agent memory
ARCHIVAL_TABLE_NAME = "letta_archival_memory_agent"  # agent memory

# external data source tables
PASSAGE_TABLE_NAME = "letta_passages"  # chunked/embedded passages (from source)
FILE_TABLE_NAME = "letta_files"  # original files (from source)


class StorageConnector:
    """Defines a DB connection that is user-specific to access data: files, Passages, Archival/Recall Memory"""

    type: Type[BaseModel]

    def __init__(
        self,
        table_type: Union[TableType.ARCHIVAL_MEMORY, TableType.RECALL_MEMORY, TableType.PASSAGES, TableType.FILES],
        config: LettaConfig,
        user_id,
        agent_id=None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.table_type = table_type

        # get object type
        if table_type == TableType.ARCHIVAL_MEMORY:
            self.type = Passage
            self.table_name = ARCHIVAL_TABLE_NAME
        elif table_type == TableType.RECALL_MEMORY:
            self.type = Message
            self.table_name = RECALL_TABLE_NAME
        elif table_type == TableType.FILES:
            self.type = FileMetadata
            self.table_name = FILE_TABLE_NAME
        elif table_type == TableType.PASSAGES:
            self.type = Passage
            self.table_name = PASSAGE_TABLE_NAME
        else:
            raise ValueError(f"Table type {table_type} not implemented")
        printd(f"Using table name {self.table_name}")

        # setup base filters for agent-specific tables
        if self.table_type == TableType.ARCHIVAL_MEMORY or self.table_type == TableType.RECALL_MEMORY:
            # agent-specific table
            assert agent_id is not None, "Agent ID must be provided for agent-specific tables"
            self.filters = {"user_id": self.user_id, "agent_id": self.agent_id}
        elif self.table_type == TableType.PASSAGES or self.table_type == TableType.FILES:
            # setup base filters for user-specific tables
            assert agent_id is None, "Agent ID must not be provided for user-specific tables"
            self.filters = {"user_id": self.user_id}
        else:
            raise ValueError(f"Table type {table_type} not implemented")

    @staticmethod
    def get_storage_connector(
        table_type: Union[TableType.ARCHIVAL_MEMORY, TableType.RECALL_MEMORY, TableType.PASSAGES, TableType.FILES],
        config: LettaConfig,
        user_id,
        agent_id=None,
    ):
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            storage_type = config.archival_storage_type
        elif table_type == TableType.RECALL_MEMORY:
            storage_type = config.recall_storage_type
        elif table_type == TableType.FILES:
            storage_type = config.metadata_storage_type
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        if storage_type == "postgres":
            from letta.agent_store.db import PostgresStorageConnector

            return PostgresStorageConnector(table_type, config, user_id, agent_id)
        elif storage_type == "chroma":
            from letta.agent_store.chroma import ChromaStorageConnector

            return ChromaStorageConnector(table_type, config, user_id, agent_id)

        elif storage_type == "qdrant":
            from letta.agent_store.qdrant import QdrantStorageConnector

            return QdrantStorageConnector(table_type, config, user_id, agent_id)
        # TODO: add back
        # elif storage_type == "lancedb":
        #    from letta.agent_store.db import LanceDBConnector

        #    return LanceDBConnector(agent_config=agent_config, table_type=table_type)

        elif storage_type == "sqlite":
            from letta.agent_store.db import SQLLiteStorageConnector

            return SQLLiteStorageConnector(table_type, config, user_id, agent_id)
        elif storage_type == "milvus":
            from letta.agent_store.milvus import MilvusStorageConnector

            return MilvusStorageConnector(table_type, config, user_id, agent_id)
        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def get_archival_storage_connector(user_id, agent_id):
        config = LettaConfig.load()
        return StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config, user_id, agent_id)

    @staticmethod
    def get_recall_storage_connector(user_id, agent_id):
        config = LettaConfig.load()
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
