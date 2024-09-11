from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from sqlalchemy import and_, asc, desc, or_, select
from sqlalchemy.sql import func
from tqdm import tqdm

from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.orm.document import Document as SQLDocument
from memgpt.orm.errors import NoResultFound
from memgpt.orm.message import Message as SQLMessage
from memgpt.orm.passage import Passage as SQLPassage
from memgpt.orm.utilities import get_db_session
from memgpt.schemas.enums import TableType
from memgpt.schemas.memgpt_base import MemGPTBase
from memgpt.schemas.passage import Passage

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from memgpt.orm.sqlalchemy_base import SqlalchemyBase as SQLBase


class SQLStorageConnector(StorageConnector):
    """Storage via SQL Alchemy"""

    engine_type: str = "sql-generic"
    SQLModel: "SQLBase" = None
    db_session: "Session" = None

    def __init__(
        self, table_type: str, config: MemGPTConfig, user_id: str, agent_id: Optional[str] = None, db_session: Optional["Session"] = None
    ):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        match table_type:
            case TableType.ARCHIVAL_MEMORY:
                self.SQLModel = SQLPassage
            case TableType.RECALL_MEMORY:
                self.SQLModel = SQLMessage
            case TableType.DOCUMENTS:
                self.SQLModel = SQLDocument
            case TableType.PASSAGES:
                self.SQLModel = SQLPassage
            case _:
                raise ValueError(f"Table type {table_type} not implemented")

        self.db_session = db_session or get_db_session()

        # self.check_db_session()

    # def check_db_session(self):
    #    from sqlalchemy import text

    #    schema = self.db_session.execute(text("show search_path")).fetchone()[0]
    #    if "postgres" not in schema:
    #        raise ValueError(f"Schema: {schema}")

    def get_filters(self, filters: Optional[Dict] = {}):
        filter_conditions = {**self.filters, **(filters or {})}
        all_filters = [getattr(self.SQLModel, key) == value for key, value in filter_conditions.items() if hasattr(self.SQLModel, key)]
        return all_filters

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000, offset=0):
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            with self.db_session as session:
                db_record_chunk = session.query(self.SQLModel).filter(*filters).offset(offset).limit(page_size).all()

                # If the chunk is empty, we've retrieved all records
                if not db_record_chunk:
                    break

                # Yield a list of Record objects converted from the chunk
                yield [record.to_pydantic() for record in db_record_chunk]

                # Increment the offset to get the next chunk in the next iteration
                offset += page_size

    def get_all_cursor(
        self,
        filters: Optional[Dict] = {},
        after: str = None,
        before: str = None,
        limit: Optional[int] = 1000,
        order_by: str = "created_at",
        reverse: bool = False,
    ):
        """Get all that returns a cursor (record.id) and records"""
        filters = self.get_filters(filters)

        # generate query
        with self.db_session as session:
            query = select(self.SQLModel).filter(*filters).limit(limit)
            # query = query.order_by(asc(self.SQLModel.id))

            # records are sorted by the order_by field first, and then by the ID if two fields are the same
            if reverse:
                query = query.order_by(desc(getattr(self.SQLModel, order_by)), asc(self.SQLModel.id))
            else:
                query = query.order_by(asc(getattr(self.SQLModel, order_by)), asc(self.SQLModel.id))

            # cursor logic: filter records based on before/after ID
            if after:
                after_value = getattr(self.get(id=after), order_by)
                sort_exp = getattr(self.SQLModel, order_by) > after_value
                query = query.filter(
                    or_(sort_exp, and_(getattr(self.SQLModel, order_by) == after_value, self.SQLModel.id > after))  # tiebreaker case
                )
            if before:
                before_value = getattr(self.get(id=before), order_by)
                sort_exp = getattr(self.SQLModel, order_by) < before_value
                query = query.filter(or_(sort_exp, and_(getattr(self.SQLModel, order_by) == before_value, self.SQLModel.id < before)))

            # get records
            db_record_chunk = session.execute(query).scalars()

            if not db_record_chunk:
                return (None, [])
            records = [record.to_pydantic() for record in db_record_chunk]
            next_cursor = db_record_chunk[-1].id
            assert isinstance(next_cursor, str)

            # return (cursor, list[records])
            return (next_cursor, records)

    def get_all(self, filters: Optional[Dict] = {}, limit=None):
        filters = self.get_filters(filters)
        with self.db_session as session:
            query = select(self.SQLModel).filter(*filters)
            if limit:
                query = query.limit(limit)
            db_records = session.execute(query).scalars()

            return [record.to_pydantic() for record in db_records]

    def get(self, id: str):
        try:
            self.check_db_session()

            db_record = self.SQLModel.read(db_session=self.db_session, identifier=id)
        except NoResultFound:
            return None

        return db_record.to_pydantic()

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        filters = self.get_filters(filters)
        with self.db_session as session:
            return session.query(self.SQLModel).filter(*filters).count()

    def insert(self, record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def insert_many(self, records, exists_ok=True, show_progress=False):
        match self.engine_type:
            case "sql-sqlite":
                from sqlalchemy.dialects.sqlite import insert
            case "sql-postgres":
                from sqlalchemy.dialects.postgresql import insert
            case _:
                from sqlalchemy.expression import insert

        if len(records) == 0:
            return
        if isinstance(records[0], Passage):
            with self.db_session as conn:
                db_records = [vars(record) for record in records]
                stmt = insert(self.SQLModel.__table__).values(db_records)
                if exists_ok:
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["id"], set_={c.name: c for c in stmt.excluded}  # Replace with your primary key column
                    )
                    conn.execute(upsert_stmt)
                else:
                    conn.execute(stmt)
                conn.commit()
        else:
            with self.db_session as session:
                iterable = tqdm(records) if show_progress else records
                # Using SQLAlchemy Core is way faster than ORM Bulk Operations https://stackoverflow.com/a/34344200
                session.execute(self.SQLModel.__table__.insert(), [vars(record) for record in iterable])
                session.commit()

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        with self.db_session as session:
            query = select(self.SQLModel).filter(*filters).order_by(self.SQLModel.embedding.l2_distance(query_vec)).limit(top_k)
            results = session.execute(query).scalars()

            return [result.to_pydantic() for result in results]

    def update(self, record: MemGPTBase):
        """Updates a record in the database based on the provided Pydantic Record object."""
        self.SQLModel(**record.model_dump(exclude_none=True)).update(self.db_session)

    def list_data_sources(self):
        assert self.table_type == TableType.ARCHIVAL_MEMORY, f"list_data_sources only implemented for ARCHIVAL_MEMORY"
        with self.db_session as session:
            unique_data_sources = session.query(self.SQLModel.data_source).filter(*self.filters).distinct().all()
            return unique_data_sources

    def query_date(self, start_date, end_date, limit=None, offset=0):
        filters = self.get_filters({})
        with self.db_session as session:
            query = (
                select(self.SQLModel)
                .filter(*filters)
                .filter(self.SQLModel.created_at >= start_date)
                .filter(self.SQLModel.created_at <= end_date)
                .filter(self.SQLModel.role != "system")
                .filter(self.SQLModel.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = session.execute(query).scalars()
            return [result.to_pydantic() for result in results]

    def query_text(self, query, limit=None, offset=0):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        filters = self.get_filters({})
        with self.db_session as session:
            query = (
                select(self.SQLModel)
                .filter(*filters)
                .filter(func.lower(self.SQLModel.text).contains(func.lower(query)))
                .filter(self.SQLModel.role != "system")
                .filter(self.SQLModel.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = session.execute(query).scalars()

            return [result.to_pydantic() for result in results]

    def delete(self, filters: Optional[Dict] = {}):
        # TODO: do we want to support soft deletes here?
        with self.db_session as session:
            session.query(self.SQLModel).filter(*self.get_filters(filters)).delete()
            session.commit()


class PostgresStorageConnector(SQLStorageConnector):
    """Storage via Postgres"""

    engine_type = "sql-postgres"

    # from pgvector.sqlalchemy import Vector
    # for c in self.SQLModel.__table__.columns:
    #     if c.name == "embedding":
    #         assert isinstance(c.type, Vector), f"Embedding column must be of type Vector, got {c.type}"

    def str_to_datetime(self, str_date: str) -> datetime:
        val = str_date.split("-")
        _datetime = datetime(int(val[0]), int(val[1]), int(val[2]))
        return _datetime

    def query_date(self, start_date, end_date, limit=None, offset=0):
        filters = self.get_filters({})
        _start_date = self.str_to_datetime(start_date) if isinstance(start_date, str) else start_date
        _end_date = self.str_to_datetime(end_date) if isinstance(end_date, str) else end_date
        with self.db_session as session:
            query = (
                select(self.SQLModel)
                .filter(*filters)
                .filter(self.SQLModel.created_at >= _start_date)
                .filter(self.SQLModel.created_at <= _end_date)
                .filter(self.SQLModel.role != "system")
                .filter(self.SQLModel.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = session.execute(query).scalars()

            return [result.to_pydantic() for result in results]


class SQLLiteStorageConnector(SQLStorageConnector):
    engine_type = "sql-sqlite"
