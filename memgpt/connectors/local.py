from typing import Optional, List, Iterator
import shutil
from memgpt.config import AgentConfig, MemGPTConfig
from tqdm import tqdm
import re
import pickle
import os

import json
import glob
from typing import List, Optional, Dict
from abc import abstractmethod

from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.empty.base import EmptyIndex
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import TextNode

from memgpt.constants import MEMGPT_DIR
from memgpt.data_types import Record
from memgpt.config import MemGPTConfig
from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.config import AgentConfig, MemGPTConfig
from memgpt.utils import printd, get_local_time, parse_formatted_time
from memgpt.data_types import Message, Passage, Record

# class VectorIndexStorageConnector(StorageConnector):

# """Local storage connector based on LlamaIndex"""

# def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
# super().__init__(table_type=table_type, agent_config=agent_config)
# config = MemGPTConfig.load()

## TODO: add asserts to avoid both being passed
# if agent_config is not None:
# self.name = agent_config.name
# self.save_directory = agent_config.save_agent_index_dir()
# else:
# self.name = name
# self.save_directory = f"{MEMGPT_DIR}/archival/{name}"

## llama index contexts
# self.embed_model = embedding_model()
# self.service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model, chunk_size=config.embedding_chunk_size)
# set_global_service_context(self.service_context)

## load/create index
# self.save_path = f"{self.save_directory}/nodes.pkl"
# if os.path.exists(self.save_path):
# self.nodes = pickle.load(open(self.save_path, "rb"))
# else:
# self.nodes = []

## create vectorindex
# if len(self.nodes):
# self.index = VectorStoreIndex(self.nodes)
# else:
# self.index = EmptyIndex()

# def load(self, filters: Dict):
## load correct version based off filters
# if "agent_id" in filters and filters["agent_id"] is not None:
## load agent archival memory
# save_directory = self.agent_config.save_agent_index_dir()
# elif "data_source" in filters and filters["data_source"] is not None:
# name = filters["data_source"]
# save_directory = f"{MEMGPT_DIR}/archival/{name}"
# else:
# raise ValueError(f"Cannot load index without agent_id or data_source {filters}")
# save_path = f"{save_directory}/nodes.pkl"
# if os.path.exists(save_path):
# nodes = pickle.load(open(save_path, "rb"))
# else:
# nodes = []
## create vectorindex
# if len(self.nodes):
# self.index = VectorStoreIndex(self.nodes)
# else:
# self.index = EmptyIndex()


# def get_nodes(self) -> List[TextNode]:
# """Get llama index nodes"""
# embed_dict = self.index._vector_store._data.embedding_dict
# node_dict = self.index._docstore.docs

# nodes = []
# for node_id, node in node_dict.items():
# vector = embed_dict[node_id]
# node.embedding = vector
# nodes.append(TextNode(text=node.text, embedding=vector))
# return nodes

# def add_nodes(self, nodes: List[TextNode]):
# self.nodes += nodes
# self.index = VectorStoreIndex(self.nodes)

# def get_all_paginated(self, page_size: int = 100) -> Iterator[List[Passage]]:
# """Get all passages in the index"""
# nodes = self.get_nodes()
# for i in tqdm(range(0, len(nodes), page_size)):
# yield [Passage(text=node.text, embedding=node.embedding) for node in nodes[i : i + page_size]]

# def get_all(self, limit: int) -> List[Passage]:
# passages = []
# for node in self.get_nodes():
# assert node.embedding is not None, f"Node embedding is None"
# passages.append(Passage(text=node.text, embedding=node.embedding))
# if len(passages) >= limit:
# break
# return passages

# def get(self, id: str) -> Passage:
# pass

# def insert(self, passage: Passage):
# nodes = [TextNode(text=passage.text, embedding=passage.embedding)]
# self.nodes += nodes
# if isinstance(self.index, EmptyIndex):
# self.index = VectorStoreIndex(self.nodes, service_context=self.service_context, show_progress=True)
# else:
# self.index.insert_nodes(nodes)

# def insert_many(self, passages: List[Passage]):
# nodes = [TextNode(text=passage.text, embedding=passage.embedding) for passage in passages]
# self.nodes += nodes
# if isinstance(self.index, EmptyIndex):
# self.index = VectorStoreIndex(self.nodes, service_context=self.service_context, show_progress=True)
# else:
# orig_size = len(self.get_nodes())
# self.index.insert_nodes(nodes)
# assert len(self.get_nodes()) == orig_size + len(
# passages
# ), f"expected {orig_size + len(passages)} nodes, got {len(self.get_nodes())} nodes"

# def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
# if isinstance(self.index, EmptyIndex):  # empty index
# return []
## TODO: this may be super slow?
## the nice thing about creating this here is that now we can save the persistent storage manager
# retriever = VectorIndexRetriever(
# index=self.index,  # does this get refreshed?
# similarity_top_k=top_k,
# )
# nodes = retriever.retrieve(query)
# results = [Passage(embedding=node.embedding, text=node.text) for node in nodes]
# return results

# def save(self):
## assert len(self.nodes) == len(self.get_nodes()), f"Expected {len(self.nodes)} nodes, got {len(self.get_nodes())} nodes"
# self.nodes = self.get_nodes()
# os.makedirs(self.save_directory, exist_ok=True)
# pickle.dump(self.nodes, open(self.save_path, "wb"))

# @staticmethod
# def list_loaded_data():
# sources = []
# for data_source_file in os.listdir(os.path.join(MEMGPT_DIR, "archival")):
# name = os.path.basename(data_source_file)
# sources.append(name)
# return sources

# def size(self):
# return len(self.get_nodes())


class InMemoryStorageConnector(StorageConnector):
    """Really dumb class so we can have a unified storae connector interface - keeps everything in memory"""

    """ Backwards compatible with previous version of recall memory """

    # TODO: maybae replace this with sqllite?

    def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
        super().__init__(table_type=table_type, agent_config=agent_config)
        config = MemGPTConfig.load()

        # supported table types
        self.supported_types = [TableType.RECALL_MEMORY]
        if table_type not in self.supported_types:
            raise ValueError(f"Table type {table_type} not supported by InMemoryStorageConnector")

        # TODO: load if exists
        self.agent_config = agent_config
        if agent_config is None:
            # is a data source
            raise ValueError("Cannot load data source from InMemoryStorageConnector")
        else:
            directory = agent_config.save_state_dir()
            if os.path.exists(directory):
                print(f"Loading saved agent {agent_config.name} from {directory}")
                json_files = glob.glob(os.path.join(directory, "*.json"))  # This will list all .json files in the current directory.
                if not json_files:
                    print(f"/load error: no .json checkpoint files found")
                    raise ValueError(f"Cannot load {agent_config.name} - no saved checkpoints found in {directory}")

                # Sort files based on modified timestamp, with the latest file being the first.
                filename = max(json_files, key=os.path.getmtime)
                state = json.load(open(filename, "r"))

                # load persistence manager
                filename = os.path.basename(filename).replace(".json", ".persistence.pickle")
                directory = agent_config.save_persistence_manager_dir()
                printd(f"Loading persistence manager from {os.path.join(directory, filename)}")
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    self.rows = data["all_messages"]
            else:
                print(f"Creating new agent {agent_config.name}")
                self.rows = []

        # convert to Record class
        self.rows = [self.json_to_message(m) for m in self.rows]

    def get_all_paginated(self, page_size: int, filters: Optional[Dict] = {}) -> Iterator[List[Record]]:
        offset = 0
        while True:
            yield self.rows[offset : offset + page_size]
            offset += page_size
            if offset >= len(self.rows):
                break

    def get_all(self, limit: Optional[int] = None, filters: Optional[Dict] = {}) -> List[Record]:
        if limit:
            return self.rows[:limit]
        return self.rows

    def get(self, id: str) -> Record:
        match_row = [row for row in self.rows if row.id == id]
        if len(match_row) == 0:
            return None
        assert len(match_row) == 1, f"Expected 1 match, got {len(match_row)} matches"
        return match_row[0]

    def insert(self, record: Record):
        self.rows.append(record)

    def insert_many(self, records: List[Record]):
        self.rows += records

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        raise NotImplementedError

    def json_to_message(self, message_json) -> Message:
        """Convert agent message JSON into Message object"""
        timestamp = message_json["timestamp"]
        message = message_json["message"]

        return Message(
            user_id=self.config.anon_clientid,
            agent_id=self.agent_config.name,
            role=message["role"],
            text=message["content"],
            model=self.agent_config.model,
            created_at=parse_formatted_time(timestamp),
            function_name=message["function_name"] if "function_name" in message else None,
            function_args=message["function_args"] if "function_args" in message else None,
            function_response=message["function_response"] if "function_response" in message else None,
            id=message["id"] if "id" in message else None,
        )

    def message_to_json(self, message: Message) -> Dict:
        """Convert Message object into JSON"""
        return {
            "timestamp": message.created_at.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            "message": {
                "role": message.role,
                "content": message.text,
                "function_name": message.function_name,
                "function_args": message.function_args,
                "function_response": message.function_response,
                "id": message.id,
            },
        }

    def save(self):
        """Save state of storage connector"""
        timestamp = get_local_time().replace(" ", "_").replace(":", "_")
        filename = f"{timestamp}.persistence.pickle"
        os.makedirs(self.config.save_persistence_manager_dir(), exist_ok=True)
        filename = os.path.join(self.config.save_persistence_manager_dir(), filename)

        all_messages = [self.message_to_json(m) for m in self.rows]

        with open(filename, "wb") as fh:
            ## TODO: fix this hacky solution to pickle the retriever
            pickle.dump(
                {
                    "all_messages": all_messages,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            printd(f"Saved state to {fh}")

    def size(self, filters: Optional[Dict] = {}) -> int:
        return len(self.rows)

    def query_date(self, start_date, end_date) -> List[Record]:
        return [row for row in self.rows if row.created_at >= start_date and row.created_at <= end_date]

    def query_text(self, query: str) -> List[Record]:
        return [row for row in self.rows if row.role not in ["system", "function"] and query.lower() in row.text.lower()]

    def delete(self, filters: Optional[Dict] = {}):
        raise NotImplementedError

    def delete_table(self, filters: Optional[Dict] = {}):
        if os.path.exists(self.agent_config.save_state_dir()):
            shutil.rmtree(self.agent_config.save_state_dir())
        if os.path.exists(self.agent_config.save_persistence_manager_dir()):
            shutil.rmtree(self.agent_config.save_persistence_manager_dir())
