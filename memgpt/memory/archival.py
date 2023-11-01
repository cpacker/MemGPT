from abc import ABC, abstractmethod
import os
import datetime
import re
import faiss
import numpy as np
from typing import Optional, List, Tuple

from memgpt.config import AgentConfig, MemGPTConfig
from .constants import MESSAGE_SUMMARY_WARNING_TOKENS, MEMGPT_DIR
from .utils import cosine_similarity, get_local_time, printd, count_tokens
from .prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM
from memgpt import utils
from .openai_tools import (
    acompletions_with_backoff as acreate,
    async_get_embedding_with_backoff,
    get_embedding_with_backoff,
    completions_with_backoff as create,
)
from llama_index import (
    VectorStoreIndex,
    EmptyIndex,
    get_response_synthesizer,
    load_index_from_storage,
    StorageContext,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

# TODO: move to different file
import psycopg2
from sqlalchemy import make_url


class ArchivalMemory(ABC):

    """Wrapper around Llama Index VectorStoreIndex"""

    @abstractmethod
    def insert(self, memory_string):
        """Insert new archival memory

        :param memory_string: Memory string to insert
        :type memory_string: str
        """
        pass

    @abstractmethod
    def search(self, query_string, count=None, start=None) -> Tuple[List[str], int]:
        """Search archival memory

        :param query_string: Query string
        :type query_string: str
        :param count: Number of results to return (None for all)
        :type count: Optional[int]
        :param start: Offset to start returning results from (None if 0)
        :type start: Optional[int]

        :return: Tuple of (list of results, total number of results)
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class LocalArchivalMemory(ArchivalMemory):
    """Archival memory built on top of Llama Index"""

    def __init__(self, agent_config, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archiva_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """

        self.top_k = top_k
        self.agent_config = agent_config

        # locate saved index
        if self.agent_config.data_source is not None:  # connected data source
            directory = f"{MEMGPT_DIR}/archival/{self.agent_config.data_source}"
            assert os.path.exists(directory), f"Archival memory database {self.agent_config.data_source} does not exist"
        elif self.agent_config.name is not None:
            directory = agent_config.save_agent_index_dir()
            if not os.path.exists(directory):
                # no existing archival storage
                directory = None

        # load/create index
        if directory:
            storage_context = StorageContext.from_defaults(persist_dir=directory)
            self.index = load_index_from_storage(storage_context)
        else:
            self.index = EmptyIndex()

        # create retriever
        if isinstance(self.index, EmptyIndex):
            self.retriever = None  # cant create retriever over empty indes
        else:
            self.retriever = VectorIndexRetriever(
                index=self.index,  # does this get refreshed?
                similarity_top_k=self.top_k,
            )

        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def save(self):
        """Save the index to disk"""
        if self.agent_config.data_source:  # update original archival index
            # TODO: this corrupts the originally loaded data. do we want to do this?
            utils.save_index(self.index, self.agent_config.data_source)
        else:
            utils.save_agent_index(self.index, self.agent_config)

    async def insert(self, memory_string):
        self.index.insert(memory_string)

        # TODO: figure out if this needs to be refreshed (probably not)
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

    async def search(self, query_string, count=None, start=None):
        if self.retriever is None:
            print("Warning: archival memory is empty")
            return [], 0

        start = start if start else 0
        count = count if count else self.top_k
        count = min(count + start, self.top_k)

        if query_string not in self.cache:
            self.cache[query_string] = self.retriever.retrieve(query_string)

        results = self.cache[query_string][start : start + count]
        results = [{"timestamp": get_local_time(), "content": node.node.text} for node in results]
        # from pprint import pprint
        # pprint(results)
        return results, len(results)

    async def a_search(self, query_string, count=None, start=None):
        return self.search(query_string, count, start)

    def __repr__(self) -> str:
        print(self.index.ref_doc_info)
        return ""


class PostgresArchivalMemory(ArchivalMemory):
    def __init__(
        self,
        agent_config: AgentConfig,
        connection_string: str,
        db_name: str,
    ):
        self.agent_config = agent_config
        self.connection_string = connection_string
        self.db_name = db_name
        self.table_name = "archival_memory"
        self.top_k = 100

        # create table
        self.conn = psycopg2.connect(self.connection_string)
        self.conn.autocommit = True

        with self.conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")

        url = make_url(connection_string)
        vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=self.table_name,
            embed_dim=MemGPTConfig.load().embedding_dim,  # openai embedding dimension
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        query_engine = index.as_query_engine()

        # create retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )


class ChromaArchivalMemory(ArchivalMemory):

    import chromadb

    def __init__(
        self,
        agent_config: AgentConfig,
        top_k: int = 100,
    ):
        self.agent_config = agent_config
        self.data_source_name = agent_config.data_source

        # connect to client
        self.client = chromadb.Client()
        # client = chromadb.PersistentClient(path="/path/to/save/to")
        self.collection = self.client.get_collection(self.data_source_name)

        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def search(self, query_string, count=None, start=None):

        start = start if start else 0
        count = count if count else self.top_k
        count = min(count + start, self.top_k)

        if query_string not in self.cache:
            self.cache[query_string] = self.collection.query(
                query_texts=[query_string],
            )

        results = self.cache[query_string][start : start + count]
        results = [{"timestamp": get_local_time(), "content": node.node.text} for node in results]
        # from pprint import pprint
        # pprint(results)
        return results, len(results)
