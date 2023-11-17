""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""
from typing import Optional, List, Iterator
import re
import pickle
import os


from typing import List, Optional
from abc import abstractmethod
import numpy as np
from tqdm import tqdm


from memgpt.config import AgentConfig, MemGPTConfig


class Passage:
    """A passage is a single unit of memory, and a standard format accross all storage backends.

    It is a string of text with an associated embedding.
    """

    def __init__(self, text: str, embedding: np.ndarray, doc_id: Optional[str] = None, passage_id: Optional[str] = None):
        self.text = text
        self.embedding = embedding
        self.doc_id = doc_id
        self.passage_id = passage_id

    def __repr__(self):
        return f"Passage(text={self.text}, embedding={self.embedding})"


class StorageConnector:
    @staticmethod
    def get_storage_connector(name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        storage_type = MemGPTConfig.load().archival_storage_type

        if storage_type == "local":
            from memgpt.connectors.local import LocalStorageConnector

            return LocalStorageConnector(name=name, agent_config=agent_config)

        elif storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector(name=name, agent_config=agent_config)

        elif storage_type == "lancedb":
            from memgpt.connectors.db import LanceDBConnector

            return LanceDBConnector(name=name)

        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def list_loaded_data():
        storage_type = MemGPTConfig.load().archival_storage_type
        if storage_type == "local":
            from memgpt.connectors.local import LocalStorageConnector

            return LocalStorageConnector.list_loaded_data()
        elif storage_type == "postgres":
            from memgpt.connectors.db import PostgresStorageConnector

            return PostgresStorageConnector.list_loaded_data()

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
