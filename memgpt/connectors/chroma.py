import chromadb
import json
import re
from typing import Optional, List, Iterator, Dict
from memgpt.connectors.storage import StorageConnector, TableType
from memgpt.utils import printd, datetime_to_timestamp, timestamp_to_datetime
from memgpt.config import AgentConfig, MemGPTConfig
from memgpt.data_types import Record, Message, Passage


class ChromaStorageConnector(StorageConnector):
    """Storage via Chroma"""

    # WARNING: This is not thread safe. Do NOT do concurrent access to the same collection.
    # Timestamps are converted to integer timestamps for chroma (datetime not supported)

    def __init__(self, table_type: str, agent_config: Optional[AgentConfig] = None):
        super().__init__(table_type=table_type, agent_config=agent_config)
        config = MemGPTConfig.load()

        # create chroma client
        if config.archival_storage_path:
            self.client = chromadb.PersistentClient(config.archival_storage_path)
        else:
            # assume uri={ip}:{port}
            ip = config.archival_storage_uri.split(":")[0]
            port = config.archival_storage_uri.split(":")[1]
            self.client = chromadb.HttpClient(host=ip, port=port)

        # get a collection or create if it doesn't exist already
        self.collection = self.client.get_or_create_collection(self.table_name)
        self.include = ["id", "documents", "embeddings", "metadatas"]

    def get_all_paginated(self, page_size: int, filters: Optional[Dict]) -> Iterator[List[Record]]:
        offset = 0
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            db_chunks = self.collection.get(offset=offset, limit=page_size, include=self.include, where=filters)

            # If the chunk is empty, we've retrieved all records
            if not db_chunks:
                break

            # Yield a list of Record objects converted from the chunk
            yield self.results_to_records(db_chunks)

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def results_to_records(self, results):
        # convert timestamps to datetime
        for metadata in results["metadatas"]:
            if "created_at" in metadata:
                metadata["created_at"] = timestamp_to_datetime(metadata["created_at"])
        return [
            self.type(id=id, text=text, embedding=embedding, **metadatas)
            for (id, text, embedding, metadatas) in zip(results["ids"], results["documents"], results["embeddings"], results["metadatas"])
        ]

    def get_all(self, limit=10, filters: Optional[Dict] = {}) -> List[Record]:
        filters = self.get_filters(filters)
        results = self.collection.get(include=self.include, where=filters)
        return self.results_to_records(results)

    def get(self, id: str, filters: Optional[Dict] = {}) -> Optional[Record]:
        filters = self.get_filters(filters)
        results = self.collection.get(ids=[id])
        return self.results_to_records(results)

    def insert(self, record: Record):
        if record.id is None:
            record.id = str(self.collection.count())
        metadata = vars(record)
        metadata.pop("id")
        metadata.pop("text")
        metadata.pop("embedding")
        self.collection.add(documents=[record.text], embeddings=[record.embedding], ids=[record.id], metadatas=[metadata])

    def insert_many(self, records: List[Record], show_progress=True):
        count = self.collection.count()
        metadatas = []
        ids = []
        documents = [record.text for record in records]
        embeddings = [record.embedding for record in records]
        for record in records:
            if record.id is None:
                count += 1
                ids.append(str(count))
                # TODO: ensure that other record.ids dont match
            metadata = vars(record)
            metadata.pop("id")
            metadata.pop("text")
            metadata.pop("embedding")
            if "created_at" in metadata:
                metadata["created_at"] = datetime_to_timestamp(metadata["created_at"])
            metadata = {key: value for key, value in metadata.items() if value is not None}  # null values not allowed
            metadatas.append(metadata)
        if not any(embeddings):
            self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        else:
            self.collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def delete(self):
        self.client.delete_collection(name=self.table_name)

    def save(self):
        # save to persistence file (nothing needs to be done)
        printd("Saving chroma")
        pass

    def size(self, filters: Optional[Dict] = {}) -> int:
        filters = self.get_filters(filters)
        return self.collection.count()

    def list_data_sources(self):
        raise NotImplementedError

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        filters = self.get_filters(filters)
        results = self.collection.query(query_embeddings=[query_vec], n_results=top_k, include=self.include, where=filters)
        return self.results_to_records(results)

    def query_date(self, start_date, end_date, start=None, count=None):
        # TODO: no idea if this is correct
        filters = self.get_filters(filters)
        filters["created_at"] = {
            "$gte": start_date,
            "$lte": end_date,
        }
        results = self.collection.query(where=filters)
        start = 0 if start is None else start
        count = len(results) if count is None else count
        results = results[start : start + count]
        return self.results_to_records(results)

    def query_text(self, query, count=None, start=None, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        results = self.collection.query(where_document={"$contains": {"text": query}}, where=filters)
        start = 0 if start is None else start
        count = len(results) if count is None else count
        results = results[start : start + count]
        return self.results_to_records(results)
