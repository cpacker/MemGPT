import os
import uuid
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, cast

from letta.agent_store.storage import StorageConnector, TableType
from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.data_types import Passage, Record, RecordType
from letta.utils import datetime_to_timestamp, timestamp_to_datetime

TEXT_PAYLOAD_KEY = "text_content"
METADATA_PAYLOAD_KEY = "metadata"


class QdrantStorageConnector(StorageConnector):
    """Storage via Qdrant"""

    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as e:
            raise ImportError("'qdrant-client' not installed. Run `pip install qdrant-client`.") from e
        assert table_type in [TableType.ARCHIVAL_MEMORY, TableType.PASSAGES], "Qdrant only supports archival memory"
        if config.archival_storage_uri and len(config.archival_storage_uri.split(":")) == 2:
            host, port = config.archival_storage_uri.split(":")
            self.qdrant_client = QdrantClient(host=host, port=port, api_key=os.getenv("QDRANT_API_KEY"))
        elif config.archival_storage_path:
            self.qdrant_client = QdrantClient(path=config.archival_storage_path)
        else:
            raise ValueError("Qdrant storage requires either a URI or a path to the storage configured")
        if not self.qdrant_client.collection_exists(self.table_name):
            self.qdrant_client.create_collection(
                collection_name=self.table_name,
                vectors_config=models.VectorParams(
                    size=MAX_EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
        self.uuid_fields = ["id", "user_id", "agent_id", "source_id", "file_id"]

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: int = 10) -> Iterator[List[RecordType]]:
        from qdrant_client import grpc

        filters = self.get_qdrant_filters(filters)
        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=self.table_name,
                scroll_filter=filters,
                limit=page_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId) and next_offset.num == 0 and next_offset.uuid == ""
            )
            yield self.to_records(results)

    def get_all(self, filters: Optional[Dict] = {}, limit=10) -> List[RecordType]:
        if self.size(filters) == 0:
            return []
        filters = self.get_qdrant_filters(filters)
        results, _ = self.qdrant_client.scroll(
            self.table_name,
            scroll_filter=filters,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        return self.to_records(results)

    def get(self, id: str) -> Optional[RecordType]:
        results = self.qdrant_client.retrieve(
            collection_name=self.table_name,
            ids=[str(id)],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        return self.to_records(results)[0]

    def insert(self, record: Record):
        points = self.to_points([record])
        self.qdrant_client.upsert(self.table_name, points=points)

    def insert_many(self, records: List[RecordType], show_progress=False):
        points = self.to_points(records)
        self.qdrant_client.upsert(self.table_name, points=points)

    def delete(self, filters: Optional[Dict] = {}):
        filters = self.get_qdrant_filters(filters)
        self.qdrant_client.delete(self.table_name, points_selector=filters)

    def delete_table(self):
        self.qdrant_client.delete_collection(self.table_name)
        self.qdrant_client.close()

    def size(self, filters: Optional[Dict] = {}) -> int:
        filters = self.get_qdrant_filters(filters)
        return self.qdrant_client.count(collection_name=self.table_name, count_filter=filters).count

    def close(self):
        self.qdrant_client.close()

    def query(
        self,
        query: str,
        query_vec: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = {},
    ) -> List[RecordType]:
        filters = self.get_filters(filters)
        results = self.qdrant_client.search(
            self.table_name,
            query_vector=query_vec,
            query_filter=filters,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        return self.to_records(results)

    def to_records(self, records: list) -> List[RecordType]:
        parsed_records = []
        for record in records:
            record = deepcopy(record)
            metadata = record.payload[METADATA_PAYLOAD_KEY]
            text = record.payload[TEXT_PAYLOAD_KEY]
            _id = metadata.pop("id")
            embedding = record.vector
            for key, value in metadata.items():
                if key in self.uuid_fields:
                    metadata[key] = uuid.UUID(value)
                elif key == "created_at":
                    metadata[key] = timestamp_to_datetime(value)
            parsed_records.append(
                cast(
                    RecordType,
                    self.type(
                        text=text,
                        embedding=embedding,
                        id=uuid.UUID(_id),
                        **metadata,
                    ),
                )
            )
        return parsed_records

    def to_points(self, records: List[RecordType]):
        from qdrant_client import models

        assert all(isinstance(r, Passage) for r in records)
        points = []
        records = list(set(records))
        for record in records:
            record = vars(record)
            _id = record.pop("id")
            text = record.pop("text", "")
            embedding = record.pop("embedding", {})
            record_metadata = record.pop("metadata_", None) or {}
            if "created_at" in record:
                record["created_at"] = datetime_to_timestamp(record["created_at"])
            metadata = {key: value for key, value in record.items() if value is not None}
            metadata = {
                **metadata,
                **record_metadata,
                "id": str(_id),
            }
            for key, value in metadata.items():
                if key in self.uuid_fields:
                    metadata[key] = str(value)
            points.append(
                models.PointStruct(
                    id=str(_id),
                    vector=embedding,
                    payload={
                        TEXT_PAYLOAD_KEY: text,
                        METADATA_PAYLOAD_KEY: metadata,
                    },
                )
            )
        return points

    def get_qdrant_filters(self, filters: Optional[Dict] = {}):
        from qdrant_client import models

        filter_conditions = {**self.filters, **filters} if filters is not None else self.filters
        must_conditions = []
        for key, value in filter_conditions.items():
            match_value = str(value) if key in self.uuid_fields else value
            field_condition = models.FieldCondition(
                key=f"{METADATA_PAYLOAD_KEY}.{key}",
                match=models.MatchValue(value=match_value),
            )
            must_conditions.append(field_condition)
        return models.Filter(must=must_conditions)
