""" Metadata store for user/agent/data_source information"""
import os
from typing import Optional, List, Dict
from memgpt.constants import DEFAULT_HUMAN, DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_PRESET, LLM_MAX_TOKENS
from memgpt.utils import get_local_time
from memgpt.data_types import AgentState, Source, User, LLMConfig, EmbeddingConfig
from memgpt.config import MemGPTConfig

from sqlalchemy import create_engine, Column, String, BIGINT, select, inspect, text, JSON, BLOB, BINARY, ARRAY, Boolean
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, mapped_column, declarative_base
from sqlalchemy.orm.session import close_all_sessions
from sqlalchemy.sql import func
from sqlalchemy import Column, BIGINT, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy_json import mutable_json_type, MutableJson
from sqlalchemy import TypeDecorator, CHAR
import uuid


from sqlalchemy.orm import sessionmaker, mapped_column, declarative_base


Base = declarative_base()

# Custom UUID type
class CommonUUID(TypeDecorator):
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR())

    def process_bind_param(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return str(value)  # Convert UUID to string for SQLite

    def process_result_value(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return uuid.UUID(value)


class LLMConfigColumn(TypeDecorator):

    """Custom type for storing LLMConfig as JSON"""

    impl = JSON

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return vars(value)
        return value

    def process_result_value(self, value, dialect):
        if value:
            return LLMConfig(**value)
        return value


class EmbeddingConfigColumn(TypeDecorator):

    """Custom type for storing EmbeddingConfig as JSON"""

    impl = JSON

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return vars(value)
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
        return value


class UserModel(Base):

    __abstract__ = True

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    default_preset = Column(String)
    default_persona = Column(String)
    default_human = Column(String)
    default_agent = Column(String)

    default_llm_config = Column(LLMConfigColumn)
    default_embedding_config = Column(EmbeddingConfigColumn)

    azure_key = Column(String, nullable=True)
    azure_endpoint = Column(String, nullable=True)
    azure_version = Column(String, nullable=True)
    azure_deployment = Column(String, nullable=True)

    openai_key = Column(String, nullable=True)
    policies_accepted = Column(Boolean, nullable=False, default=False)

    def __repr__(self):
        return f"<User(id='{self.id}')>"

    def to_record(self):
        return User(
            id=self.id,
            default_preset=self.default_preset,
            default_persona=self.default_persona,
            default_human=self.default_human,
            default_agent=self.default_agent,
            default_llm_config=self.default_llm_config,
            default_embedding_config=self.default_embedding_config,
            azure_key=self.azure_key,
            azure_endpoint=self.azure_endpoint,
            azure_version=self.azure_version,
            azure_deployment=self.azure_deployment,
            openai_key=self.openai_key,
            policies_accepted=self.policies_accepted,
        )


class AgentModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __abstract__ = True

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    persona_file = Column(String)
    human_file = Column(String)
    preset = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    memgpt_version = Column(String)

    # configs
    llm_config = Column(LLMConfigColumn)
    embedding_config = Column(EmbeddingConfigColumn)

    # state
    state = Column(JSON)
    attached_source_ids = Column(ARRAY(CommonUUID))

    def __repr__(self):
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self):
        return AgentState(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            persona_file=self.persona_file,
            human_file=self.human_file,
            preset=self.preset,
            created_at=self.created_at,
            memgpt_version=self.memgpt_version,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            state=self.state,
            attached_source_ids=self.attached_source_ids,
        )


class SourceModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __abstract__ = True  # this line is necessary

    # Assuming passage_id is the primary key
    # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Source(passage_id='{self.id}', name='{self.name}')>"

    def to_record(self):
        return Source(id=self.id, user_id=self.user_id, name=self.name, created_at=self.created_at)


class MetadataStore:
    def __init__(self, config: MemGPTConfig):

        # TODO: get DB URI or path
        if config.metadata_storage_type == "postgres":
            self.uri = config.metadata_storage_uri
        elif config.metadata_storage_type == "sqlite":
            path = os.path.join(config.metadata_storage_path, "sqlite.db")
            self.uri = f"sqlite:///{path}"
        else:
            raise ValueError(f"Invalid metadata storage type: {config.metadata_storage_type}")

        # TODO: check to see if table(s) need to be greated or not

        self.engine = create_engine(self.uri)
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)

    def create_agent(self, agent: AgentState):
        # insert into agent table
        session = self.Session()
        session.add(AgentModel(**vars(agent)))
        session.commit()

    def create_source(self, source: Source):
        session = self.Session()
        session.add(SourceModel(**vars(source)))
        session.commit()

    def create_user(self, user: User):
        session = self.Session()
        session.add(UserModel(**vars(user)))
        session.commit()

    def update_agent(self, agent: AgentState):
        session = self.Session()
        session.query(AgentModel).filter(AgentModel.id == agent.id).update(vars(agent))
        session.commit()

    def update_user(self, user: User):
        session = self.Session()
        session.query(UserModel).filter(UserModel.id == user.id).update(vars(user))
        session.commit()

    def update_source(self, source: Source):
        session = self.Session()
        session.query(SourceModel).filter(SourceModel.id == source.id).update(vars(source))
        session.commit()

    def delete_agent(self, agent_id: str):
        session = self.Session()
        session.query(AgentModel).filter(AgentModel.id == agent_id).delete()

    def delete_source(self, source_id: str):
        pass

    def delete_user(self, user_id: str):
        pass

    def list_agents(self, user_id):
        session = self.Session()
        results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
        return [r.to_record() for r in results]

    def list_sources(self, user_id):
        session = self.Session()
        results = session.query(SourceModel).filter(SourceModel.user_id == user_id).all()
        return [r.to_record() for r in results]

    def get_agent(self, agent_id):
        pass

    def get_user(self, user_id):
        pass

    def get_source(self, source_id):
        pass
