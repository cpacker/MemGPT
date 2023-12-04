""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""
from typing import Any, Optional, List, Iterator
import re
import pickle
import os


from typing import List, Optional
from abc import abstractmethod
import numpy as np
from tqdm import tqdm


from memgpt.config import AgentConfig, MemGPTConfig


# ENUM representing table types in MemGPT
class TableType:
    ARCHIVAL_MEMORY = "archival_memory"  # recall memory table: memgpt_agent_{agent_id}
    RECALL_MEMORY = "recall_memory"  # archival memory table: memgpt_agent_recall_{agent_id}
    DOCUMENTS = "documents"
    USERS = "users"
    AGENTS = "agents"


# Defining schema objects:
# Note: user/agent can borrow from MemGPTConfig/AgentConfig classes


class Record:
    """Base class for all table records"""

    def __init__(self, user_id: str):
        self.user_id = user_id


class Message(Record):
    """A message is a single unit of communication, and a standard format accross all storage backends.

    It is a string of text with an associated embedding.
    """

    def __init__(
        self, user_id: str, agent_id: str, role: str, text: str, embedding: Optional[np.ndarray] = None, message_id: Optional[str] = None
    ):
        super().__init__(user_id)
        self.agent_id = agent_id
        self.role = role
        self.text = text
        self.embedding = embedding  # optional
        self.message_id = message_id
        self.message_type = "conversation"
        # todo: self.role = role (?)

    def __repr__(self):
        return f"Message(text={self.text}, embedding={self.embedding})"


class FunctionCallMessage(Message):
    def __init__(
        self, user_id: str, agent_id: str, role: str, text: str, embedding: Optional[np.ndarray] = None, message_id: Optional[str] = None
    ):
        super().__init__(user_id, agent_id, role, text, None, message_id)
        self.message_type = "function"

        # todo: self.role = role (?)

    def __repr__(self):
        return f"Message(text={self.text}, embedding={self.embedding})"


class Document(Record):
    """A document represent a document loaded into MemGPT, which is broken down into passages."""

    def __init__(self, user_id: str, text: str, document_id: Optional[str] = None):
        super().__init__(user_id)
        self.text = text
        self.document_id = document_id
        # TODO: add optional embedding?

    def __repr__(self) -> str:
        pass


class Passage(Record):
    """A passage is a single unit of memory, and a standard format accross all storage backends.

    It is a string of text with an associated embedding.
    """

    def __init__(self, user_id: str, text: str, embedding: np.ndarray, doc_id: Optional[str] = None, passage_id: Optional[str] = None):
        super().__init__(user_id)
        self.text = text
        self.embedding = embedding
        self.doc_id = doc_id
        self.passage_id = passage_id

    def __repr__(self):
        return f"Passage(text={self.text}, embedding={self.embedding})"


class StorageConnector:
    @staticmethod
    def get_archival_storage_connector(name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        storage_type = MemGPTConfig.load().archival_storage_type

        if storage_type == "local":
            from memgpt.connectors.local import VectorIndexStorageConnector

            return VectorIndexStorageConnector(name=name, agent_config=agent_config)

        elif storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector(name=name, agent_config=agent_config)
        elif storage_type == "chroma":
            from memgpt.connectors.chroma import ChromaStorageConnector

            return ChromaStorageConnector(name=name, agent_config=agent_config)
        elif storage_type == "lancedb":
            from memgpt.connectors.db import LanceDBConnector

            return LanceDBConnector(name=name, agent_config=agent_config)
        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def get_recall_storage_connector(name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        storage_type = MemGPTConfig.load().recall_storage_type

        if storage_type == "local":
            from memgpt.connectors.local import VectorIndexStorageConnector

            # maintains in-memory list for storage
            return InMemoryStorageConnector(name=name, agent_config=agent_config)

        elif storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector(name=name, agent_config=agent_config)

        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def list_loaded_data():
        storage_type = MemGPTConfig.load().archival_storage_type
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
    def get_all_paginated(self, page_size: int) -> Iterator[List[Passage]]:
        pass

    @abstractmethod
    def get_all(self, limit: int) -> List[Passage]:
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
    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        pass

    @abstractmethod
    def save(self):
        """Save state of storage connector"""
        pass

    @abstractmethod
    def size(self):
        """Get number of passages (text/embedding pairs) in storage"""
        pass
