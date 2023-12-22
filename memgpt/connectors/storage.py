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
from memgpt.data_types import Record, Passage, Document, Message
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


# table names used by MemGPT
RECALL_TABLE_NAME = "memgpt_recall_memory_agent"  # agent memory
ARCHIVAL_TABLE_NAME = "memgpt_archival_memory_agent"  # agent memory
PASSAGE_TABLE_NAME = "memgpt_passages"  # loads data sources
DOCUMENT_TABLE_NAME = "memgpt_documents"


class StorageConnector:
    def __init__(self, table_type: TableType, agent_config: Optional[AgentConfig] = None):

        config = MemGPTConfig.load()
        self.agent_config = agent_config
        self.user_id = config.anon_clientid
        self.table_type = table_type

        # get object type
        if table_type == TableType.ARCHIVAL_MEMORY:
            self.type = Passage
        elif table_type == TableType.RECALL_MEMORY:
            self.type = Message
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        # determine name of database table
        self.table_name = self.generate_table_name(agent_config, table_type=table_type)
        printd(f"Using table name {self.table_name}")

        # setup base filters
        if self.table_type == TableType.ARCHIVAL_MEMORY or self.table_type == TableType.RECALL_MEMORY:
            # agent-specific table
            self.filters = {"user_id": self.user_id, "agent_id": self.agent_config.name}
        else:
            self.filters = {"user_id": self.user_id}

    def get_filters(self, filters: Optional[Dict] = {}):
        # get all filters for query
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        print("FILTERS", filter_conditions)
        return filter_conditions

    def generate_table_name(self, agent_config: AgentConfig, table_type: TableType):

        if agent_config is not None:
            # Table names for agent-specific tables
            if agent_config.memgpt_version < "0.2.6":
                # if agent is prior version, use old table name
                if table_type == TableType.ARCHIVAL_MEMORY:
                    return f"memgpt_agent_{self.sanitize_table_name(agent_config.name)}"
                else:
                    raise ValueError(f"Table type {table_type} not implemented")
            else:
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
            else:
                raise ValueError(f"Table type {table_type} not implemented")

    @staticmethod
    def get_storage_connector(table_type: TableType, storage_type: Optional[str] = None, agent_config: Optional[AgentConfig] = None):

        # read from config if not provided
        if storage_type is None:
            storage_type = MemGPTConfig.load().archival_storage_type

        if storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector(agent_config=agent_config, table_type=table_type)
        elif storage_type == "chroma":
            from memgpt.connectors.chroma import ChromaStorageConnector

            return ChromaStorageConnector(agent_config=agent_config, table_type=table_type)
        elif storage_type == "lancedb":
            from memgpt.connectors.db import LanceDBConnector

            return LanceDBConnector(agent_config=agent_config, table_type=table_type)

        elif storage_type == "local":
            from memgpt.connectors.local import InMemoryStorageConnector

            return InMemoryStorageConnector(agent_config=agent_config, table_type=table_type)

        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def get_archival_storage_connector(agent_config: Optional[AgentConfig] = None):
        return StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, agent_config=agent_config)

    @staticmethod
    def get_recall_storage_connector(agent_config: Optional[AgentConfig] = None):
        return StorageConnector.get_storage_connector(TableType.RECALL_MEMORY, agent_config=agent_config)

    @staticmethod
    def list_loaded_data(storage_type: Optional[str] = None):
        # TODO: modify this to simply list loaded data from a given user
        if storage_type is None:
            storage_type = MemGPTConfig.load().archival_storage_type

            return

        if storage_type == "local":
            from memgpt.connectors.local import VectorIndexStorageConnector

            return VectorIndexStorageConnector.list_loaded_data()
        elif storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector.list_loaded_data()
        elif storage_type == "chroma":
            from memgpt.connectors.chroma import ChromaStorageConnector

            return ChromaStorageConnector.list_loaded_data()
        elif storage_type == "lancedb":
            from memgpt.connectors.db import LanceDBConnector

            return LanceDBConnector.list_loaded_data()
        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @abstractmethod
    def get_all_paginated(self, page_size: int, filters: Optional[Dict] = {}) -> Iterator[List[Record]]:
        pass

    @abstractmethod
    def get_all(self, limit: int, filters: Optional[Dict]) -> List[Record]:
        pass

    @abstractmethod
    def get(self, id: str) -> Record:
        pass

    @abstractmethod
    def insert(self, record: Record):
        pass

    @abstractmethod
    def insert_many(self, records: List[Record]):
        pass

    @abstractmethod
    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        pass

    @abstractmethod
    def save(self):
        """Save state of storage connector"""
        pass

    @abstractmethod
    def size(self, filters: Optional[Dict] = {}) -> int:
        """Get number of passages (text/embedding pairs) in storage"""
        pass
