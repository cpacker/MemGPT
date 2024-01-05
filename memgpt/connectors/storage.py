""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""
from typing import Any, Optional, List, Iterator
import re
import pickle
import os
from abc import abstractmethod

from typing import List, Optional, Dict
from tqdm import tqdm


from memgpt.config import AgentConfig, MemGPTConfig
from memgpt.data_types import Record, Passage, Document, Message, Source
from memgpt.utils import printd


# ENUM representing table types in MemGPT
# each table corresponds to a different table schema  (specified in data_types.py)
class TableType:
    ARCHIVAL_MEMORY = "archival_memory"  # recall memory table: memgpt_agent_{agent_id}
    RECALL_MEMORY = "recall_memory"  # archival memory table: memgpt_agent_recall_{agent_id}
    PASSAGES = "passages"  # TODO
    DOCUMENTS = "documents"  # TODO
    USERS = "users"  # TODO
    AGENTS = "agents"  # TODO
    DATA_SOURCES = "data_sources"  # TODO


# table names used by MemGPT

# agent tables
RECALL_TABLE_NAME = "memgpt_recall_memory_agent"  # agent memory
ARCHIVAL_TABLE_NAME = "memgpt_archival_memory_agent"  # agent memory

# external data source tables
SOURCE_TABLE_NAME = "memgpt_sources"  # metadata for loaded data source
PASSAGE_TABLE_NAME = "memgpt_passages"  # chunked/embedded passages (from source)
DOCUMENT_TABLE_NAME = "memgpt_documents"  # original documents (from source)


class StorageConnector:
    def __init__(self, table_type: TableType, agent_config: Optional[AgentConfig] = None):
        config = MemGPTConfig.load()
        self.agent_config = agent_config
        self.user_id = config.anon_clientid
        self.table_type = table_type

        # get object type
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            self.type = Passage
        elif table_type == TableType.RECALL_MEMORY:
            self.type = Message
        elif table_type == TableType.DATA_SOURCES:
            self.type = Source
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        # determine name of database table
        self.table_name = self.generate_table_name(agent_config, table_type=table_type)
        printd(f"Using table name {self.table_name}")

        # setup base filters for agent-specific tables
        if self.table_type == TableType.ARCHIVAL_MEMORY or self.table_type == TableType.RECALL_MEMORY:
            # agent-specific table
            self.filters = {"user_id": self.user_id, "agent_id": self.agent_config.name}
        elif self.table_type == TableType.PASSAGES or self.table_type == TableType.DOCUMENTS or self.table_type == TableType.DATA_SOURCES:
            # setup base filters for user-specific tables
            self.filters = {"user_id": self.user_id}
        else:
            self.filters = {}

    def get_filters(self, filters: Optional[Dict] = {}):
        # get all filters for query
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        return filter_conditions

    def generate_table_name(self, agent_config: AgentConfig, table_type: TableType):
        if agent_config is not None:
            # Table names for agent-specific tables
            if table_type == TableType.ARCHIVAL_MEMORY:
                return ARCHIVAL_TABLE_NAME
            elif table_type == TableType.RECALL_MEMORY:
                return RECALL_TABLE_NAME
            else:
                raise ValueError(f"Table type {table_type} not implemented")
        else:
            # table names for non-agent specific tables
            if table_type == TableType.PASSAGES:
                return PASSAGE_TABLE_NAME
            elif table_type == TableType.DOCUMENTS:
                return DOCUMENT_TABLE_NAME
            elif table_type == TableType.DATA_SOURCES:
                return SOURCE_TABLE_NAME
            else:
                raise ValueError(f"Table type {table_type} not implemented")

    @staticmethod
    def get_storage_connector(table_type: TableType, storage_type: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        # read from config if not provided
        if storage_type is None:
            if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
                storage_type = MemGPTConfig.load().archival_storage_type
            elif table_type == TableType.RECALL_MEMORY:
                storage_type = MemGPTConfig.load().recall_storage_type
            elif table_type == TableType.DATA_SOURCES or table_type == TableType.USERS or table_type == TableType.AGENTS:
                storage_type = MemGPTConfig.load().metadata_storage_type
            # TODO: other tables

        if storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector(agent_config=agent_config, table_type=table_type)
        elif storage_type == "chroma":
            from memgpt.connectors.chroma import ChromaStorageConnector

            return ChromaStorageConnector(agent_config=agent_config, table_type=table_type)

        # TODO: add back
        # elif storage_type == "lancedb":
        #    from memgpt.connectors.db import LanceDBConnector

        #    return LanceDBConnector(agent_config=agent_config, table_type=table_type)

        elif storage_type == "local":
            from memgpt.connectors.local import InMemoryStorageConnector

            return InMemoryStorageConnector(agent_config=agent_config, table_type=table_type)

        elif storage_type == "sqlite":
            from memgpt.connectors.db import SQLLiteStorageConnector

            return SQLLiteStorageConnector(agent_config=agent_config, table_type=table_type)

        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def get_archival_storage_connector(agent_config: Optional[AgentConfig] = None):
        return StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, agent_config=agent_config)

    @staticmethod
    def get_recall_storage_connector(agent_config: Optional[AgentConfig] = None):
        return StorageConnector.get_storage_connector(TableType.RECALL_MEMORY, agent_config=agent_config)

    @staticmethod
    def get_metadata_storage_connector(table_type: TableType):
        storage_type = MemGPTConfig.load().metadata_storage_type
        return StorageConnector.get_storage_connector(table_type, storage_type=storage_type)

    @abstractmethod
    def get_filters(self, filters: Optional[Dict] = {}):
        pass

    @abstractmethod
    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000) -> Iterator[List[Record]]:
        pass

    @abstractmethod
    def get_all(self, filters: Optional[Dict] = {}, limit=10) -> List[Record]:
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[Record]:
        pass

    @abstractmethod
    def size(self, filters: Optional[Dict] = {}) -> int:
        pass

    @abstractmethod
    def insert(self, record: Record):
        pass

    @abstractmethod
    def insert_many(self, records: List[Record], show_progress=False):
        pass

    @abstractmethod
    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
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
