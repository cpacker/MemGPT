import chromadb
import json
import re
from typing import Optional, List, Iterator
from memgpt.connectors.storage import StorageConnector, Passage
from memgpt.utils import printd
from memgpt.config import AgentConfig, MemGPTConfig


def create_chroma_client():
    config = MemGPTConfig.load()
    # create chroma client
    if config.archival_storage_path:
        client = chromadb.PersistentClient(config.archival_storage_path)
    else:
        # assume uri={ip}:{port}
        ip = config.archival_storage_uri.split(":")[0]
        port = config.archival_storage_uri.split(":")[1]
        client = chromadb.HttpClient(host=ip, port=port)
    return client


class ChromaStorageConnector(StorageConnector):
    """Storage via Chroma"""

    # WARNING: This is not thread safe. Do NOT do concurrent access to the same collection.

    def __init__(self, name: Optional[str] = None, agent_config: Optional[AgentConfig] = None):
        # determine table name
        if agent_config:
            assert name is None, f"Cannot specify both agent config and name {name}"
            self.table_name = self.generate_table_name_agent(agent_config)
        elif name:
            assert agent_config is None, f"Cannot specify both agent config and name {name}"
            self.table_name = self.generate_table_name(name)
        else:
            raise ValueError("Must specify either agent config or name")

        printd(f"Using table name {self.table_name}")

        # create client
        self.client = create_chroma_client()

        # get a collection or create if it doesn't exist already
        self.collection = self.client.get_or_create_collection(self.table_name)

    def get_all_paginated(self, page_size: int) -> Iterator[List[Passage]]:
        offset = 0
        while True:
            # Retrieve a chunk of records with the given page_size
            db_passages_chunk = self.collection.get(offset=offset, limit=page_size, include=["embeddings", "documents"])

            # If the chunk is empty, we've retrieved all records
            if not db_passages_chunk:
                break

            # Yield a list of Passage objects converted from the chunk
            yield [Passage(text=p.text, embedding=p.embedding, doc_id=p.doc_id, passage_id=p.id) for p in db_passages_chunk]

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self) -> List[Passage]:
        results = self.collection.get(include=["embeddings", "documents"])
        return [Passage(text=text, embedding=embedding) for (text, embedding) in zip(results["documents"], results["embeddings"])]

    def get(self, id: str) -> Optional[Passage]:
        results = self.collection.get(ids=[id])
        return [Passage(text=text, embedding=embedding) for (text, embedding) in zip(results["documents"], results["embeddings"])]

    def insert(self, passage: Passage):
        self.collection.add(documents=[passage.text], embeddings=[passage.embedding], ids=[str(self.collection.count())])

    def insert_many(self, passages: List[Passage], show_progress=True):
        count = self.collection.count()
        ids = [str(count + i) for i in range(len(passages))]
        self.collection.add(
            documents=[passage.text for passage in passages], embeddings=[passage.embedding for passage in passages], ids=ids
        )

    def query(self, query: str, query_vec: List[float], top_k: int = 10) -> List[Passage]:
        results = self.collection.query(query_embeddings=[query_vec], n_results=top_k, include=["embeddings", "documents"])
        # get index [0] since query is passed as list
        return [Passage(text=text, embedding=embedding) for (text, embedding) in zip(results["documents"][0], results["embeddings"][0])]

    def delete(self):
        self.client.delete_collection(name=self.table_name)

    def save(self):
        # save to persistence file (nothing needs to be done)
        printd("Saving chroma")
        pass

    @staticmethod
    def list_loaded_data():
        client = create_chroma_client()
        collections = client.list_collections()
        collections = [c.name for c in collections if c.name.startswith("memgpt_") and not c.name.startswith("memgpt_agent_")]
        return collections

    def sanitize_table_name(self, name: str) -> str:
        # Remove leading and trailing whitespace
        name = name.strip()

        # Replace spaces and invalid characters with underscores
        name = re.sub(r"\s+|\W+", "_", name)

        # Truncate to the maximum identifier length (e.g., 63 for PostgreSQL)
        max_length = 63
        if len(name) > max_length:
            name = name[:max_length].rstrip("_")

        # Convert to lowercase
        name = name.lower()

        return name

    def generate_table_name_agent(self, agent_config: AgentConfig):
        return f"memgpt_agent_{self.sanitize_table_name(agent_config.name)}"

    def generate_table_name(self, name: str):
        return f"memgpt_{self.sanitize_table_name(name)}"

    def size(self) -> int:
        return self.collection.count()
