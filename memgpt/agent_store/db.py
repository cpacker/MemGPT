from sqlalchemy import create_engine, Column, String, BIGINT, select, text, JSON, DateTime
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, mapped_column, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy_json import MutableJson
from sqlalchemy import TypeDecorator, CHAR
import uuid

from tqdm import tqdm
from typing import Optional, List, Dict
from tqdm import tqdm

from memgpt.config import MemGPTConfig
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.config import MemGPTConfig
from memgpt.data_types import Record, Message, Passage, ToolCall, RecordType
from memgpt.constants import MAX_EMBEDDING_DIM


# Custom UUID type
class CommonUUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR())

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


# Custom serialization / de-serialization for JSON columns


class ToolCallColumn(TypeDecorator):
    """Custom type for storing List[ToolCall] as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return [vars(v) for v in value]
        return value

    def process_result_value(self, value, dialect):
        if value:
            return [ToolCall(**v) for v in value]
        return value


Base = declarative_base()


def get_db_model(
    table_name: str,
    table_type: TableType,
    dialect="postgresql",
):
    # Define a helper function to create or get the model class
    def create_or_get_model(class_name, base_model, table_name):
        if class_name in globals():
            return globals()[class_name]
        Model = type(class_name, (base_model,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        globals()[class_name] = Model
        return Model

    if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
        # create schema for archival memory
        class PassageModel(Base):
            """Defines data model for storing Passages (consisting of text, embedding)"""

            __abstract__ = True  # this line is necessary

            # Assuming passage_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(CommonUUID, nullable=False)
            text = Column(String)
            doc_id = Column(CommonUUID)
            agent_id = Column(CommonUUID)
            data_source = Column(String)  # agent_name if agent, data_source name if from data source

            # vector storage

            from pgvector.sqlalchemy import Vector

            embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
            embedding_dim = Column(BIGINT)
            embedding_model = Column(String)

            metadata_ = Column(MutableJson)
            created_at = Column(DateTime(timezone=True), server_default=func.now())

            def __repr__(self):
                return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

            def to_record(self):
                return Passage(
                    text=self.text,
                    embedding=self.embedding,
                    embedding_dim=self.embedding_dim,
                    embedding_model=self.embedding_model,
                    doc_id=self.doc_id,
                    user_id=self.user_id,
                    id=self.id,
                    data_source=self.data_source,
                    agent_id=self.agent_id,
                    metadata_=self.metadata_,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model" + dialect
        return create_or_get_model(class_name, PassageModel, table_name)

    elif table_type == TableType.RECALL_MEMORY:

        class MessageModel(Base):
            """Defines data model for storing Message objects"""

            __abstract__ = True  # this line is necessary

            # Assuming message_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(CommonUUID, nullable=False)
            agent_id = Column(CommonUUID, nullable=False)

            # openai info
            role = Column(String, nullable=False)
            text = Column(String)  # optional: can be null if function call
            model = Column(String)  # optional: can be null if LLM backend doesn't require specifying
            name = Column(String)  # optional: multi-agent only

            # tool call request info
            # if role == "assistant", this MAY be specified
            # if role != "assistant", this must be null
            # TODO align with OpenAI spec of multiple tool calls
            tool_calls = Column(ToolCallColumn)

            # tool call response info
            # if role == "tool", then this must be specified
            # if role != "tool", this must be null
            tool_call_id = Column(String)

            # vector storage
            from pgvector.sqlalchemy import Vector

            embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
            embedding_dim = Column(BIGINT)
            embedding_model = Column(String)

            # Add a datetime column, with default value as the current time
            created_at = Column(DateTime(timezone=True), server_default=func.now())

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
                    tool_calls=self.tool_calls,
                    tool_call_id=self.tool_call_id,
                    embedding=self.embedding,
                    embedding_dim=self.embedding_dim,
                    embedding_model=self.embedding_model,
                    created_at=self.created_at,
                    id=self.id,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model" + dialect
        return create_or_get_model(class_name, MessageModel, table_name)

    else:
        raise ValueError(f"Table type {table_type} not implemented")


class SQLStorageConnector(StorageConnector):
    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        self.config = config

    def get_filters(self, filters: Optional[Dict] = {}):
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        all_filters = [getattr(self.db_model, key) == value for key, value in filter_conditions.items()]
        return all_filters

    def get_all(self, filters: Optional[Dict] = {}, limit=None) -> List[RecordType]:
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            if limit:
                db_records = session.query(self.db_model).filter(*filters).limit(limit).all()
            else:
                db_records = session.query(self.db_model).filter(*filters).all()
        return [record.to_record() for record in db_records]

    def get(self, id: uuid.UUID) -> Optional[Record]:
        with self.session_maker() as session:
            db_record = session.get(self.db_model, id)
        if db_record is None:
            return None
        return db_record.to_record()

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            return session.query(self.db_model).filter(*filters).count()

    def insert(self, record: Record):
        raise NotImplementedError

    def insert_many(self, records: List[RecordType], show_progress=False):
        raise NotImplementedError

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[RecordType]:
        raise NotImplementedError("Vector query not implemented for SQLStorageConnector")

    def save(self):
        return

    def query_date(self, start_date, end_date, offset=0, limit=None):
        filters = self.get_filters({})
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(self.db_model.created_at >= start_date)
                .filter(self.db_model.created_at <= end_date)
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return [result.to_record() for result in results]

    def query_text(self, query, offset=0, limit=None):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        filters = self.get_filters({})
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(func.lower(self.db_model.text).contains(func.lower(query)))
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        # return [self.type(**vars(result)) for result in results]
        return [result.to_record() for result in results]

    def delete(self, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            session.query(self.db_model).filter(*filters).delete()
            session.commit()


class PostgresStorageConnector(SQLStorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        from pgvector.sqlalchemy import Vector

        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        # get storage URI
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            self.uri = self.config.archival_storage_uri
            if self.config.archival_storage_uri is None:
                raise ValueError(f"Must specifiy archival_storage_uri in config {self.config.config_path}")
        elif table_type == TableType.RECALL_MEMORY:
            self.uri = self.config.recall_storage_uri
            if self.config.recall_storage_uri is None:
                raise ValueError(f"Must specifiy recall_storage_uri in config {self.config.config_path}")
        else:
            raise ValueError(f"Table type {table_type} not implemented")
        # create table
        self.db_model = get_db_model(self.table_name, table_type)
        self.engine = create_engine(self.uri)
        for c in self.db_model.__table__.columns:
            if c.name == "embedding":
                assert isinstance(c.type, Vector), f"Embedding column must be of type Vector, got {c.type}"

        Base.metadata.create_all(self.engine, tables=[self.db_model.__table__])  # Create the table if it doesn't exist

        self.session_maker = sessionmaker(bind=self.engine)
        with self.session_maker() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[RecordType]:
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            results = session.scalars(
                select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)
            ).all()

        # Convert the results into Passage objects
        records = [result.to_record() for result in results]
        return records

    def insert_many(self, records: List[RecordType], exists_ok=True, show_progress=False):
        from sqlalchemy.dialects.postgresql import insert

        # TODO: this is terrible, should eventually be done the same way for all types (migrate to SQLModel)
        if len(records) == 0:
            return
        if isinstance(records[0], Passage):
            with self.engine.connect() as conn:
                db_records = [vars(record) for record in records]
                # print("records", db_records)
                stmt = insert(self.db_model.__table__).values(db_records)
                # print(stmt)
                if exists_ok:
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["id"], set_={c.name: c for c in stmt.excluded}  # Replace with your primary key column
                    )
                    print(upsert_stmt)
                    conn.execute(upsert_stmt)
                else:
                    conn.execute(stmt)
                conn.commit()
        else:
            with self.session_maker() as session:
                iterable = tqdm(records) if show_progress else records
                for record in iterable:
                    db_record = self.db_model(**vars(record))
                    session.add(db_record)
                session.commit()

    def insert(self, record: Record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def update(self, record: RecordType):
        """
        Updates a record in the database based on the provided Record object.
        """
        with self.session_maker() as session:
            # Find the record by its ID
            db_record = session.query(self.db_model).filter_by(id=record.id).first()
            if not db_record:
                raise ValueError(f"Record with id {record.id} does not exist.")

            # Update the record with new values from the provided Record object
            for attr, value in vars(record).items():
                setattr(db_record, attr, value)

            # Commit the changes to the database
            session.commit()
