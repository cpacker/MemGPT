from typing import Optional, List, Iterator
from memgpt.config import AgentConfig, MemGPTConfig
from tqdm import tqdm
import re
import pickle
import os


from typing import List, Optional

from llama_index import VectorStoreIndex, EmptyIndex, ServiceContext, set_global_service_context
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import TextNode

from memgpt.constants import MEMGPT_DIR
from memgpt.config import MemGPTConfig
from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.config import AgentConfig, MemGPTConfig


class LocalStorageConnector(StorageConnector):

    """Local storage connector based on LlamaIndex"""

    def __init__(self, name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        from memgpt.embeddings import embedding_model

        config = MemGPTConfig.load()

        # TODO: add asserts to avoid both being passed
        if name is None:
            self.name = agent_config.name
            self.save_directory = agent_config.save_agent_index_dir()
        else:
            self.name = name
            self.save_directory = f"{MEMGPT_DIR}/archival/{name}"

        # llama index contexts
        self.embed_model = embedding_model()
        self.service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model, chunk_size=config.embedding_chunk_size)
        set_global_service_context(self.service_context)

        # load/create index
        self.save_path = f"{self.save_directory}/nodes.pkl"
        if os.path.exists(self.save_path):
            self.nodes = pickle.load(open(self.save_path, "rb"))
        else:
            self.nodes = []

        # create vectorindex
        if len(self.nodes):
            self.index = VectorStoreIndex(self.nodes)
        else:
            self.index = EmptyIndex()

    def get_nodes(self) -> List[TextNode]:
        """Get llama index nodes"""
        embed_dict = self.index._vector_store._data.embedding_dict
        node_dict = self.index._docstore.docs

        nodes = []
        for node_id, node in node_dict.items():
            vector = embed_dict[node_id]
            node.embedding = vector
            nodes.append(TextNode(text=node.text, embedding=vector))
        return nodes

    def add_nodes(self, nodes: List[TextNode]):
        self.nodes += nodes
        self.index = VectorStoreIndex(self.nodes)

    def get_all_paginated(self, page_size: int = 100) -> Iterator[List[Passage]]:
        """Get all passages in the index"""
        nodes = self.get_nodes()
        for i in tqdm(range(0, len(nodes), page_size)):
            yield [Passage(text=node.text, embedding=node.embedding) for node in nodes[i : i + page_size]]

    def get_all(self, limit: int) -> List[Passage]:
        passages = []
        for node in self.get_nodes():
            assert node.embedding is not None, f"Node embedding is None"
            passages.append(Passage(text=node.text, embedding=node.embedding))
            if len(passages) >= limit:
                break
        return passages

    def get(self, id: str) -> Passage:
        pass

    def insert(self, passage: Passage):
        nodes = [TextNode(text=passage.text, embedding=passage.embedding)]
        self.nodes += nodes
        if isinstance(self.index, EmptyIndex):
            self.index = VectorStoreIndex(self.nodes, service_context=self.service_context, show_progress=True)
        else:
            self.index.insert_nodes(nodes)

    def insert_many(self, passages: List[Passage]):
        nodes = [TextNode(text=passage.text, embedding=passage.embedding) for passage in passages]
        self.nodes += nodes
        if isinstance(self.index, EmptyIndex):
            self.index = VectorStoreIndex(self.nodes, service_context=self.service_context, show_progress=True)
        else:
            orig_size = len(self.get_nodes())
            self.index.insert_nodes(nodes)
            assert len(self.get_nodes()) == orig_size + len(
                passages
            ), f"expected {orig_size + len(passages)} nodes, got {len(self.get_nodes())} nodes"

    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        if isinstance(self.index, EmptyIndex):  # empty index
            return []
        # TODO: this may be super slow?
        # the nice thing about creating this here is that now we can save the persistent storage manager
        retriever = VectorIndexRetriever(
            index=self.index,  # does this get refreshed?
            similarity_top_k=top_k,
        )
        nodes = retriever.retrieve(query)
        results = [Passage(embedding=node.embedding, text=node.text) for node in nodes]
        return results

    def save(self):
        # assert len(self.nodes) == len(self.get_nodes()), f"Expected {len(self.nodes)} nodes, got {len(self.get_nodes())} nodes"
        self.nodes = self.get_nodes()
        os.makedirs(self.save_directory, exist_ok=True)
        pickle.dump(self.nodes, open(self.save_path, "wb"))

    @staticmethod
    def list_loaded_data():
        sources = []
        for data_source_file in os.listdir(os.path.join(MEMGPT_DIR, "archival")):
            name = os.path.basename(data_source_file)
            sources.append(name)
        return sources

    def size(self):
        return len(self.get_nodes())
