# type: ignore

import lancedb
import uuid
from datetime import datetime
from tqdm import tqdm
from typing import Optional, List, Iterator, Dict

from memgpt.config import MemGPTConfig
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.config import AgentConfig, MemGPTConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.utils import printd
from memgpt.data_types import Record, Message, Passage, Source

from datetime import datetime

from lancedb.pydantic import Vector, LanceModel

""" Initial implementation - not complete """


def get_db_model(table_name: str, table_type: TableType):
    config = MemGPTConfig.load()

    if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
        # create schema for archival memory
        class PassageModel(LanceModel):
            """Defines data model for storing Passages (consisting of text, embedding)"""

            id: uuid.UUID
            user_id: str
            text: str
            doc_id: str
            agent_id: str
            data_source: str
            embedding: Vector(config.default_embedding_config.embedding_dim)
            metadata_: Dict

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

        return PassageModel
    elif table_type == TableType.RECALL_MEMORY:

        class MessageModel(LanceModel):
            """Defines data model for storing Message objects"""

            __abstract__ = True  # this line is necessary

            # Assuming message_id is the primary key
            id: uuid.UUID
            user_id: str
            agent_id: str

            # openai info
            role: str
            name: str
            text: str
            model: str
            user: str

            # function info
            function_name: str
            function_args: str
            function_response: str

            embedding = Vector(config.default_embedding_config.embedding_dim)

            # Add a datetime column, with default value as the current time
            created_at = datetime

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
                    function_name=self.function_name,
                    function_args=self.function_args,
                    function_response=self.function_response,
                    embedding=self.embedding,
                    created_at=self.created_at,
                    id=self.id,
                )

        """Create database model for table_name"""
        return MessageModel

    else:
        raise ValueError(f"Table type {table_type} not implemented")


class LanceDBConnector(StorageConnector):
    """Storage via LanceDB"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        # TODO
        pass

    def generate_where_filter(self, filters: Dict) -> str:
        where_filters = []
        for key, value in filters.items():
            where_filters.append(f"{key}={value}")
        return where_filters.join(" AND ")

    @abstractmethod
    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000) -> Iterator[List[Record]]:
        # TODO
        pass

    @abstractmethod
    def get_all(self, filters: Optional[Dict] = {}, limit=10) -> List[Record]:
        # TODO
        pass

    @abstractmethod
    def get(self, id: uuid.UUID) -> Optional[Record]:
        # TODO
        pass

    @abstractmethod
    def size(self, filters: Optional[Dict] = {}) -> int:
        # TODO
        pass

    @abstractmethod
    def insert(self, record: Record):
        # TODO
        pass

    @abstractmethod
    def insert_many(self, records: List[Record], show_progress=False):
        # TODO
        pass

    @abstractmethod
    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        # TODO
        pass

    @abstractmethod
    def query_date(self, start_date, end_date):
        # TODO
        pass

    @abstractmethod
    def query_text(self, query):
        # TODO
        pass

    @abstractmethod
    def delete_table(self):
        # TODO
        pass

    @abstractmethod
    def delete(self, filters: Optional[Dict] = {}):
        # TODO
        pass

    @abstractmethod
    def save(self):
        # TODO
        pass
