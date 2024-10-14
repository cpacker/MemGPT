import uuid
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, cast

from pymilvus import DataType, MilvusClient
from pymilvus.client.constants import ConsistencyLevel

from letta.agent_store.storage import StorageConnector, TableType
from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.data_types import Passage, Record, RecordType
from letta.utils import datetime_to_timestamp, printd, timestamp_to_datetime


class MilvusStorageConnector(StorageConnector):
    """Storage via Milvus"""

    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        assert table_type in [TableType.ARCHIVAL_MEMORY, TableType.PASSAGES], "Milvus only supports archival memory"
        if config.archival_storage_uri:
            self.client = MilvusClient(uri=config.archival_storage_uri)
            self._create_collection()
        else:
            raise ValueError("Please set `archival_storage_uri` in the config file when using Milvus.")

        # need to be converted to strings
        self.uuid_fields = ["id", "user_id", "agent_id", "source_id", "file_id"]

    def _create_collection(self):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=65_535)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, is_primary=False, max_length=65_535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=MAX_EMBEDDING_DIM)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="id")
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="IP")
        self.client.create_collection(
            collection_name=self.table_name, schema=schema, index_params=index_params, consistency_level=ConsistencyLevel.Strong
        )

    def get_milvus_filter(self, filters: Optional[Dict] = {}) -> str:
        filter_conditions = {**self.filters, **filters} if filters is not None else self.filters
        if not filter_conditions:
            return ""
        conditions = []
        for key, value in filter_conditions.items():
            if key in self.uuid_fields or isinstance(key, str):
                condition = f'({key} == "{value}")'
            else:
                condition = f"({key} == {value})"
            conditions.append(condition)
        filter_expr = " and ".join(conditions)
        if len(conditions) == 1:
            filter_expr = filter_expr[1:-1]
        return filter_expr

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: int = 1000) -> Iterator[List[RecordType]]:
        if not self.client.has_collection(collection_name=self.table_name):
            yield []
        filter_expr = self.get_milvus_filter(filters)
        offset = 0
        while True:
            # Retrieve a chunk of records with the given page_size
            query_res = self.client.query(
                collection_name=self.table_name,
                filter=filter_expr,
                offset=offset,
                limit=page_size,
            )
            if not query_res:
                break
            # Yield a list of Record objects converted from the chunk
            yield self._list_to_records(query_res)

            # Increment the offset to get the next chunk in the next iteration
            offset += page_size

    def get_all(self, filters: Optional[Dict] = {}, limit=None) -> List[RecordType]:
        if not self.client.has_collection(collection_name=self.table_name):
            return []
        filter_expr = self.get_milvus_filter(filters)
        query_res = self.client.query(
            collection_name=self.table_name,
            filter=filter_expr,
            limit=limit,
        )
        return self._list_to_records(query_res)

    def get(self, id: str) -> Optional[RecordType]:
        res = self.client.get(collection_name=self.table_name, ids=str(id))
        return self._list_to_records(res)[0] if res else None

    def size(self, filters: Optional[Dict] = {}) -> int:
        if not self.client.has_collection(collection_name=self.table_name):
            return 0
        filter_expr = self.get_milvus_filter(filters)
        count_expr = "count(*)"
        query_res = self.client.query(
            collection_name=self.table_name,
            filter=filter_expr,
            output_fields=[count_expr],
        )
        doc_num = query_res[0][count_expr]
        return doc_num

    def insert(self, record: RecordType):
        self.insert_many([record])

    def insert_many(self, records: List[RecordType], show_progress=False):
        if not records:
            return

        # Milvus lite currently does not support upsert, so we delete and insert instead
        # self.client.upsert(collection_name=self.table_name, data=self._records_to_list(records))
        ids = [str(record.id) for record in records]
        self.client.delete(collection_name=self.table_name, ids=ids)
        data = self._records_to_list(records)
        self.client.insert(collection_name=self.table_name, data=data)

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[RecordType]:
        if not self.client.has_collection(self.table_name):
            return []
        search_res = self.client.search(
            collection_name=self.table_name, data=[query_vec], filter=self.get_milvus_filter(filters), limit=top_k, output_fields=["*"]
        )[0]
        entity_res = [res["entity"] for res in search_res]
        return self._list_to_records(entity_res)

    def delete_table(self):
        self.client.drop_collection(collection_name=self.table_name)

    def delete(self, filters: Optional[Dict] = {}):
        if not self.client.has_collection(collection_name=self.table_name):
            return
        filter_expr = self.get_milvus_filter(filters)
        self.client.delete(collection_name=self.table_name, filter=filter_expr)

    def save(self):
        # save to persistence file (nothing needs to be done)
        printd("Saving milvus")

    def _records_to_list(self, records: List[Record]) -> List[Dict]:
        if records == []:
            return []
        assert all(isinstance(r, Passage) for r in records)
        record_list = []
        records = list(set(records))
        for record in records:
            record_vars = deepcopy(vars(record))
            _id = record_vars.pop("id")
            text = record_vars.pop("text", "")
            embedding = record_vars.pop("embedding")
            record_metadata = record_vars.pop("metadata_", None) or {}
            if "created_at" in record_vars:
                record_vars["created_at"] = datetime_to_timestamp(record_vars["created_at"])
            record_dict = {key: value for key, value in record_vars.items() if value is not None}
            record_dict = {
                **record_dict,
                **record_metadata,
                "id": str(_id),
                "text": text,
                "embedding": embedding,
            }
            for key, value in record_dict.items():
                if key in self.uuid_fields:
                    record_dict[key] = str(value)
            record_list.append(record_dict)
        return record_list

    def _list_to_records(self, query_res: List[Dict]) -> List[RecordType]:
        records = []
        for res_dict in query_res:
            _id = res_dict.pop("id")
            embedding = res_dict.pop("embedding")
            text = res_dict.pop("text")
            metadata = deepcopy(res_dict)
            for key, value in metadata.items():
                if key in self.uuid_fields:
                    metadata[key] = uuid.UUID(value)
                elif key == "created_at":
                    metadata[key] = timestamp_to_datetime(value)
            records.append(
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
        return records
