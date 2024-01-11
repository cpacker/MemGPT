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
    cache_ok = True

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
    cache_ok = True

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
    cache_ok = True

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
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

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

    def __repr__(self) -> str:
        return f"<User(id='{self.id}')>"

    def to_record(self) -> User:
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

    __tablename__ = "agents"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    persona = Column(String)
    human = Column(String)
    preset = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # configs
    llm_config = Column(LLMConfigColumn)
    embedding_config = Column(EmbeddingConfigColumn)

    # state
    state = Column(JSON)

    def __repr__(self) -> str:
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> AgentState:
        return AgentState(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            persona=self.persona,
            human=self.human,
            preset=self.preset,
            created_at=self.created_at,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            state=self.state,
        )


class SourceModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "sources"
    __table_args__ = {"extend_existing": True}

    # Assuming passage_id is the primary key
    # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # TODO: add num passages

    def __repr__(self) -> str:
        return f"<Source(passage_id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Source:
        return Source(id=self.id, user_id=self.user_id, name=self.name, created_at=self.created_at)


class AgentSourceMappingModel(Base):

    """Stores mapping between agent -> source"""

    __tablename__ = "agent_source_mapping"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    agent_id = Column(CommonUUID, nullable=False)
    source_id = Column(CommonUUID, nullable=False)

    def __repr__(self) -> str:
        return f"<AgentSourceMapping(user_id='{self.user_id}', agent_id='{self.agent_id}', source_id='{self.source_id}')>"


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
        Base.metadata.create_all(
            self.engine, tables=[UserModel.__table__, AgentModel.__table__, SourceModel.__table__, AgentSourceMappingModel.__table__]
        )
        self.Session = sessionmaker(bind=self.engine)

    def create_agent(self, agent: AgentState):
        # insert into agent table
        session = self.Session()
        # make sure agent.name does not already exist for user user_id
        if session.query(AgentModel).filter(AgentModel.name == agent.name).filter(AgentModel.user_id == agent.user_id).count() > 0:
            raise ValueError(f"Agent with name {agent.name} already exists")
        session.add(AgentModel(**vars(agent)))
        session.commit()

    def create_source(self, source: Source):
        session = self.Session()
        # make sure source.name does not already exist for user
        if session.query(SourceModel).filter(SourceModel.name == source.name).filter(SourceModel.user_id == source.user_id).count() > 0:
            raise ValueError(f"Source with name {source.name} already exists")
        session.add(SourceModel(**vars(source)))
        session.commit()

    def create_user(self, user: User):
        session = self.Session()
        if session.query(UserModel).filter(UserModel.id == user.id).count() > 0:
            raise ValueError(f"User with id {user.id} already exists")
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
        session.commit()

    def delete_source(self, source_id: str):
        session = self.Session()

        # delete from sources table
        session.query(SourceModel).filter(SourceModel.id == source_id).delete()

        # delete any mappings
        session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.source_id == source_id).delete()

        session.commit()

    def delete_user(self, user_id: str):
        session = self.Session()

        # delete from users table
        session.query(UserModel).filter(UserModel.id == user_id).delete()

        # delete associated agents
        session.query(AgentModel).filter(AgentModel.user_id == user_id).delete()

        # delete associated sources
        session.query(SourceModel).filter(SourceModel.user_id == user_id).delete()

        # delete associated mappings
        session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.user_id == user_id).delete()

        session.commit()

    def list_agents(self, user_id: str) -> List[AgentState]:
        session = self.Session()
        results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
        return [r.to_record() for r in results]

    def list_sources(self, user_id: str) -> List[Source]:
        session = self.Session()
        results = session.query(SourceModel).filter(SourceModel.user_id == user_id).all()
        return [r.to_record() for r in results]

    def get_agent(self, agent_id: str = None, agent_name: str = None, user_id: str = None) -> Optional[AgentState]:
        session = self.Session()
        if agent_id:
            results = session.query(AgentModel).filter(AgentModel.id == agent_id).all()
        else:
            assert agent_name is not None and user_id is not None, "Must provide either agent_id or agent_name"
            results = session.query(AgentModel).filter(AgentModel.name == agent_name).filter(AgentModel.user_id == user_id).all()

        if len(results) == 0:
            return None
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"  # should only be one result
        return results[0].to_record()

    def get_user(self, user_id: str) -> Optional[User]:
        session = self.Session()
        results = session.query(UserModel).filter(UserModel.id == user_id).all()
        if len(results) == 0:
            return None
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        return results[0].to_record()

    def get_source(self, source_id: str = None, user_id: str = None, source_name: str = None) -> Optional[Source]:
        session = self.Session()
        if source_id:
            results = session.query(SourceModel).filter(SourceModel.id == source_id).all()
        else:
            assert user_id is not None and source_name is not None
            results = session.query(SourceModel).filter(SourceModel.name == source_name).filter(SourceModel.user_id == user_id).all()
        if len(results) == 0:
            return None
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        return results[0].to_record()

    # agent source metadata
    def attach_source(self, user_id: str, agent_id: str, source_id: str):
        session = self.Session()
        session.add(AgentSourceMappingModel(user_id=user_id, agent_id=agent_id, source_id=source_id))
        session.commit()

    def list_attached_sources(self, agent_id: str) -> List[Column]:
        session = self.Session()
        results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.agent_id == agent_id).all()
        return [r.source_id for r in results]

    def list_attached_agents(self, source_id):
        session = self.Session()
        results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.source_id == source_id).all()
        return [r.agent_id for r in results]

    def detach_source(self, agent_id: str, source_id: str):
        session = self.Session()
        session.query(AgentSourceMappingModel).filter(
            AgentSourceMappingModel.agent_id == agent_id, AgentSourceMappingModel.source_id == source_id
        ).delete()
        session.commit()
