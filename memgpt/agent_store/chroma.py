import chromadb
import uuid
import json
import re
from typing import Optional, List, Iterator, Dict
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.utils import printd, datetime_to_timestamp, timestamp_to_datetime
from memgpt.config import MemGPTConfig
from memgpt.data_types import Record, Message, Passage


class ChromaStorageConnector(StorageConnector):
    """Storage via Chroma"""

    # WARNING: This is not thread safe. Do NOT do concurrent access to the same collection.
    # Timestamps are converted to integer timestamps for chroma (datetime not supported)

    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        assert table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES, "Chroma only supports archival memory"

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
        self.include = ["documents", "embeddings", "metadatas"]

        # need to be converted to strings
        self.uuid_fields = ["id", "user_id", "agent_id", "source_id"]

    def get_filters(self, filters: Optional[Dict] = {}):
        # get all filters for query
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters

        # convert to chroma format
        chroma_filters = []
        ids = []
        for key, value in filter_conditions.items():
            # filter by id
            if key == "id":
                ids = [str(value)]
                continue

            # filter by other keys
            if key in self.uuid_fields:
                chroma_filters.append({key: {"$eq": str(value)}})
            else:
                chroma_filters.append({key: {"$eq": value}})

        if len(chroma_filters) > 1:
            chroma_filters = {"$and": chroma_filters}
        elif len(chroma_filters) == 0:
            chroma_filters = {}
        else:
            chroma_filters = chroma_filters[0]
        return ids, chroma_filters

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000) -> Iterator[List[Record]]:
        offset = 0
        ids, filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            results = self.collection.get(ids=ids, offset=offset, limit=page_size, include=self.include, where=filters)

            # If the chunk is empty, we've retrieved all records
            if len(results["embeddings"]) == 0:
                break

            # Yield a list of Record objects converted from the chunk
            yield self.results_to_records(results)

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def results_to_records(self, results):
        # convert timestamps to datetime
        for metadata in results["metadatas"]:
            if "created_at" in metadata:
                metadata["created_at"] = timestamp_to_datetime(metadata["created_at"])
            for key, value in metadata.items():
                if key in self.uuid_fields:
                    metadata[key] = uuid.UUID(value)
        if results["embeddings"]:  # may not be returned, depending on table type
            return [
                self.type(text=text, embedding=embedding, id=uuid.UUID(record_id), **metadatas)
                for (text, record_id, embedding, metadatas) in zip(
                    results["documents"], results["ids"], results["embeddings"], results["metadatas"]
                )
            ]
        else:
            # no embeddings
            return [
                self.type(text=text, id=uuid.UUID(id), **metadatas)
                for (text, id, metadatas) in zip(results["documents"], results["ids"], results["metadatas"])
            ]

    def get_all(self, filters: Optional[Dict] = {}, limit=None) -> List[Record]:
        ids, filters = self.get_filters(filters)
        if self.collection.count() == 0:
            return []
        if limit:
            results = self.collection.get(ids=ids, include=self.include, where=filters, limit=limit)
        else:
            results = self.collection.get(ids=ids, include=self.include, where=filters)
        return self.results_to_records(results)

    def get(self, id: str) -> Optional[Record]:
        results = self.collection.get(ids=[str(id)])
        if len(results["ids"]) == 0:
            return None
        return self.results_to_records(results)[0]

    def format_records(self, records: List[Record]):
        metadatas = []
        ids = [str(record.id) for record in records]
        documents = [record.text for record in records]
        embeddings = [record.embedding for record in records]

        # collect/format record metadata
        for record in records:
            metadata = vars(record)
            metadata.pop("id")
            metadata.pop("text")
            metadata.pop("embedding")
            if "created_at" in metadata:
                metadata["created_at"] = datetime_to_timestamp(metadata["created_at"])
            if "metadata" in metadata and metadata["metadata"] is not None:
                record_metadata = dict(metadata["metadata"])
                metadata.pop("metadata")
            else:
                record_metadata = {}
            metadata = {key: value for key, value in metadata.items() if value is not None}  # null values not allowed
            metadata = {**metadata, **record_metadata}  # merge with metadata

            # convert uuids to strings
            for key, value in metadata.items():
                if key in self.uuid_fields:
                    metadata[key] = str(value)
            metadatas.append(metadata)
        return ids, documents, embeddings, metadatas

    def insert(self, record: Record):
        ids, documents, embeddings, metadatas = self.format_records([record])
        if not any(embeddings):
            raise ValueError("Embeddings must be provided to chroma")
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def insert_many(self, records: List[Record], show_progress=False):
        ids, documents, embeddings, metadatas = self.format_records(records)
        if not any(embeddings):
            raise ValueError("Embeddings must be provided to chroma")
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def delete(self, filters: Optional[Dict] = {}):
        ids, filters = self.get_filters(filters)
        self.collection.delete(ids=ids, where=filters)

    def delete_table(self):
        # drop collection
        self.client.delete_collection(self.collection.name)

    def save(self):
        # save to persistence file (nothing needs to be done)
        printd("Saving chroma")

    def size(self, filters: Optional[Dict] = {}) -> int:
        # unfortuantely, need to use pagination to get filtering
        # warning: poor performance for large datasets
        return len(self.get_all(filters=filters))

    def list_data_sources(self):
        raise NotImplementedError

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        ids, filters = self.get_filters(filters)
        results = self.collection.query(query_embeddings=[query_vec], n_results=top_k, include=self.include, where=filters)

        # flatten, since we only have one query vector
        flattened_results = {}
        for key, value in results.items():
            if value:
                flattened_results[key] = value[0]
                assert len(value) == 1, f"Value is size {len(value)}: {value}"
            else:
                flattened_results[key] = value

        return self.results_to_records(flattened_results)

    def query_date(self, start_date, end_date, start=None, count=None):
        raise ValueError("Cannot run query_date with chroma")
        # filters = self.get_filters(filters)
        # filters["created_at"] = {
        #    "$gte": start_date,
        #    "$lte": end_date,
        # }
        # results = self.collection.query(where=filters)
        # start = 0 if start is None else start
        # count = len(results) if count is None else count
        # results = results[start : start + count]
        # return self.results_to_records(results)

    def query_text(self, query, count=None, start=None, filters: Optional[Dict] = {}):
        raise ValueError("Cannot run query_text with chroma")
        # filters = self.get_filters(filters)
        # results = self.collection.query(where_document={"$contains": {"text": query}}, where=filters)
        # start = 0 if start is None else start
        # count = len(results) if count is None else count
        # results = results[start : start + count]
        # return self.results_to_records(results)
