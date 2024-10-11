""" Metadata store for user/agent/data_source information"""

import base64
import os
import secrets
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import (
    BIGINT,
    BINARY,
    JSON,
    Boolean,
    Column,
    DateTime,
    Index,
    String,
    TypeDecorator,
    and_,
    asc,
    desc,
    or_,
    select,
    text,
)
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm.session import close_all_sessions
from sqlalchemy.sql import func
from sqlalchemy_json import MutableJson
from tqdm import tqdm

from letta.agent_store.storage import StorageConnector, TableType
from letta.base import Base
from letta.config import LettaConfig
from letta.constants import MAX_EMBEDDING_DIM
from letta.schemas.agent import AgentState
from letta.schemas.api_key import APIKey
from letta.schemas.block import Block, Human, Persona
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import JobStatus
from letta.schemas.job import Job
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import Memory

# from letta.schemas.message import Message, Passage, Record, RecordType, ToolCall
from letta.schemas.message import Message
from letta.schemas.openai.chat_completions import ToolCall, ToolCallFunction
from letta.schemas.organization import Organization
from letta.schemas.passage import Passage
from letta.schemas.source import Source
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.settings import settings
from letta.utils import enforce_types, get_utc_time, printd

config = LettaConfig()


class LLMConfigColumn(TypeDecorator):
    """Custom type for storing LLMConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            # return vars(value)
            if isinstance(value, LLMConfig):
                return value.model_dump()
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
            # return vars(value)
            if isinstance(value, EmbeddingConfig):
                return value.model_dump()
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
        return value


class ToolCallColumn(TypeDecorator):

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            values = []
            for v in value:
                if isinstance(v, ToolCall):
                    values.append(v.model_dump())
                else:
                    values.append(v)
            return values

        return value

    def process_result_value(self, value, dialect):
        if value:
            tools = []
            for tool_value in value:
                if "function" in tool_value:
                    tool_call_function = ToolCallFunction(**tool_value["function"])
                    del tool_value["function"]
                else:
                    tool_call_function = None
                tools.append(ToolCall(function=tool_call_function, **tool_value))
            return tools
        return value


class UserModel(Base):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True)
    org_id = Column(String)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True))

    # TODO: what is this?
    policies_accepted = Column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<User(id='{self.id}' name='{self.name}')>"

    def to_record(self) -> User:
        return User(id=self.id, name=self.name, created_at=self.created_at, org_id=self.org_id)


class OrganizationModel(Base):
    __tablename__ = "organizations"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<Organization(id='{self.id}' name='{self.name}')>"

    def to_record(self) -> Organization:
        return Organization(id=self.id, name=self.name, created_at=self.created_at)


# TODO: eventually store providers?
# class Provider(Base):
#    __tablename__ = "providers"
#    __table_args__ = {"extend_existing": True}
#
#    id = Column(String, primary_key=True)
#    name = Column(String, nullable=False)
#    created_at = Column(DateTime(timezone=True))
#    api_key = Column(String, nullable=False)
#    base_url = Column(String, nullable=False)


class APIKeyModel(Base):
    """Data model for authentication tokens. One-to-many relationship with UserModel (1 User - N tokens)."""

    __tablename__ = "tokens"

    id = Column(String, primary_key=True)
    # each api key is tied to a user account (that it validates access for)
    user_id = Column(String, nullable=False)
    # the api key
    key = Column(String, nullable=False)
    # extra (optional) metadata
    name = Column(String)

    Index(__tablename__ + "_idx_user", user_id),
    Index(__tablename__ + "_idx_key", key),

    def __repr__(self) -> str:
        return f"<APIKey(id='{self.id}', key='{self.key}', name='{self.name}')>"

    def to_record(self) -> User:
        return APIKey(
            id=self.id,
            user_id=self.user_id,
            key=self.key,
            name=self.name,
        )


def generate_api_key(prefix="sk-", length=51) -> str:
    # Generate 'length // 2' bytes because each byte becomes two hex digits. Adjust length for prefix.
    actual_length = max(length - len(prefix), 1) // 2  # Ensure at least 1 byte is generated
    random_bytes = secrets.token_bytes(actual_length)
    new_key = prefix + random_bytes.hex()
    return new_key


class AgentModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "agents"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(String)

    # state (context compilation)
    message_ids = Column(JSON)
    memory = Column(JSON)
    system = Column(String)
    tools = Column(JSON)

    # configs
    agent_type = Column(String)
    llm_config = Column(LLMConfigColumn)
    embedding_config = Column(EmbeddingConfigColumn)

    # state
    metadata_ = Column(JSON)

    # tools
    tools = Column(JSON)

    Index(__tablename__ + "_idx_user", user_id),

    def __repr__(self) -> str:
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> AgentState:
        return AgentState(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            created_at=self.created_at,
            description=self.description,
            message_ids=self.message_ids,
            memory=Memory.load(self.memory),  # load dictionary
            system=self.system,
            tools=self.tools,
            agent_type=self.agent_type,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            metadata_=self.metadata_,
        )


class SourceModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "sources"
    __table_args__ = {"extend_existing": True}

    # Assuming passage_id is the primary key
    # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    embedding_config = Column(EmbeddingConfigColumn)
    description = Column(String)
    metadata_ = Column(JSON)
    Index(__tablename__ + "_idx_user", user_id),

    # TODO: add num passages

    def __repr__(self) -> str:
        return f"<Source(passage_id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Source:
        return Source(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            created_at=self.created_at,
            embedding_config=self.embedding_config,
            description=self.description,
            metadata_=self.metadata_,
        )


class AgentSourceMappingModel(Base):
    """Stores mapping between agent -> source"""

    __tablename__ = "agent_source_mapping"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)
    source_id = Column(String, nullable=False)
    Index(__tablename__ + "_idx_user", user_id, agent_id, source_id),

    def __repr__(self) -> str:
        return f"<AgentSourceMapping(user_id='{self.user_id}', agent_id='{self.agent_id}', source_id='{self.source_id}')>"


class BlockModel(Base):
    __tablename__ = "block"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True, nullable=False)
    value = Column(String, nullable=False)
    limit = Column(BIGINT)
    name = Column(String, nullable=False)
    template = Column(Boolean, default=False)  # True: listed as possible human/persona
    label = Column(String)
    metadata_ = Column(JSON)
    description = Column(String)
    user_id = Column(String)
    Index(__tablename__ + "_idx_user", user_id),

    def __repr__(self) -> str:
        return f"<Block(id='{self.id}', name='{self.name}', template='{self.template}', label='{self.label}', user_id='{self.user_id}')>"

    def to_record(self) -> Block:
        if self.label == "persona":
            return Persona(
                id=self.id,
                value=self.value,
                limit=self.limit,
                name=self.name,
                template=self.template,
                label=self.label,
                metadata_=self.metadata_,
                description=self.description,
                user_id=self.user_id,
            )
        elif self.label == "human":
            return Human(
                id=self.id,
                value=self.value,
                limit=self.limit,
                name=self.name,
                template=self.template,
                label=self.label,
                metadata_=self.metadata_,
                description=self.description,
                user_id=self.user_id,
            )
        else:
            return Block(
                id=self.id,
                value=self.value,
                limit=self.limit,
                name=self.name,
                template=self.template,
                label=self.label,
                metadata_=self.metadata_,
                description=self.description,
                user_id=self.user_id,
            )


class ToolModel(Base):
    __tablename__ = "tools"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(String)
    description = Column(String)
    source_type = Column(String)
    source_code = Column(String)
    json_schema = Column(JSON)
    module = Column(String)
    tags = Column(JSON)

    def __repr__(self) -> str:
        return f"<Tool(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Tool:
        return Tool(
            id=self.id,
            name=self.name,
            user_id=self.user_id,
            description=self.description,
            source_type=self.source_type,
            source_code=self.source_code,
            json_schema=self.json_schema,
            module=self.module,
            tags=self.tags,
        )


class JobModel(Base):
    __tablename__ = "jobs"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True)
    user_id = Column(String)
    status = Column(String, default=JobStatus.pending)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata_ = Column(JSON)

    def __repr__(self) -> str:
        return f"<Job(id='{self.id}', status='{self.status}')>"

    def to_record(self):
        return Job(
            id=self.id,
            user_id=self.user_id,
            status=self.status,
            created_at=self.created_at,
            completed_at=self.completed_at,
            metadata_=self.metadata_,
        )


class MetadataStore:
    uri: Optional[str] = None

    def __init__(self, config: LettaConfig):
        # TODO: get DB URI or path
        if config.metadata_storage_type == "postgres":
            # construct URI from enviornment variables
            self.uri = settings.pg_uri if settings.pg_uri else config.metadata_storage_uri

        elif config.metadata_storage_type == "sqlite":
            path = os.path.join(config.metadata_storage_path, "sqlite.db")
            self.uri = f"sqlite:///{path}"
        else:
            raise ValueError(f"Invalid metadata storage type: {config.metadata_storage_type}")

        # Ensure valid URI
        assert self.uri, "Database URI is not provided or is invalid."

        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_api_key(self, user_id: str, name: str) -> APIKey:
        """Create an API key for a user"""
        new_api_key = generate_api_key()
        with self.session_maker() as session:
            if session.query(APIKeyModel).filter(APIKeyModel.key == new_api_key).count() > 0:
                # NOTE duplicate API keys / tokens should never happen, but if it does don't allow it
                raise ValueError(f"Token {new_api_key} already exists")
            # TODO store the API keys as hashed
            assert user_id and name, "User ID and name must be provided"
            token = APIKey(user_id=user_id, key=new_api_key, name=name)
            session.add(APIKeyModel(**vars(token)))
            session.commit()
        return self.get_api_key(api_key=new_api_key)

    @enforce_types
    def delete_api_key(self, api_key: str):
        """Delete an API key from the database"""
        with self.session_maker() as session:
            session.query(APIKeyModel).filter(APIKeyModel.key == api_key).delete()
            session.commit()

    @enforce_types
    def get_api_key(self, api_key: str) -> Optional[APIKey]:
        with self.session_maker() as session:
            results = session.query(APIKeyModel).filter(APIKeyModel.key == api_key).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"  # should only be one result
            return results[0].to_record()

    @enforce_types
    def get_all_api_keys_for_user(self, user_id: str) -> List[APIKey]:
        with self.session_maker() as session:
            results = session.query(APIKeyModel).filter(APIKeyModel.user_id == user_id).all()
            tokens = [r.to_record() for r in results]
            return tokens

    @enforce_types
    def get_user_from_api_key(self, api_key: str) -> Optional[User]:
        """Get the user associated with a given API key"""
        token = self.get_api_key(api_key=api_key)
        if token is None:
            raise ValueError(f"Provided token does not exist")
        else:
            return self.get_user(user_id=token.user_id)

    @enforce_types
    def create_agent(self, agent: AgentState):
        # insert into agent table
        # make sure agent.name does not already exist for user user_id
        with self.session_maker() as session:
            if session.query(AgentModel).filter(AgentModel.name == agent.name).filter(AgentModel.user_id == agent.user_id).count() > 0:
                raise ValueError(f"Agent with name {agent.name} already exists")
            fields = vars(agent)
            fields["memory"] = agent.memory.to_dict()
            session.add(AgentModel(**fields))
            session.commit()

    @enforce_types
    def create_source(self, source: Source):
        with self.session_maker() as session:
            if session.query(SourceModel).filter(SourceModel.name == source.name).filter(SourceModel.user_id == source.user_id).count() > 0:
                raise ValueError(f"Source with name {source.name} already exists for user {source.user_id}")
            session.add(SourceModel(**vars(source)))
            session.commit()

    @enforce_types
    def create_user(self, user: User):
        with self.session_maker() as session:
            if session.query(UserModel).filter(UserModel.id == user.id).count() > 0:
                raise ValueError(f"User with id {user.id} already exists")
            session.add(UserModel(**vars(user)))
            session.commit()

    @enforce_types
    def create_organization(self, organization: Organization):
        with self.session_maker() as session:
            if session.query(OrganizationModel).filter(OrganizationModel.id == organization.id).count() > 0:
                raise ValueError(f"Organization with id {organization.id} already exists")
            session.add(OrganizationModel(**vars(organization)))
            session.commit()

    @enforce_types
    def create_block(self, block: Block):
        with self.session_maker() as session:
            # TODO: fix?
            # we are only validating that more than one template block
            # with a given name doesn't exist.
            if (
                session.query(BlockModel)
                .filter(BlockModel.name == block.name)
                .filter(BlockModel.user_id == block.user_id)
                .filter(BlockModel.template == True)
                .filter(BlockModel.label == block.label)
                .count()
                > 0
            ):

                raise ValueError(f"Block with name {block.name} already exists")
            session.add(BlockModel(**vars(block)))
            session.commit()

    @enforce_types
    def create_tool(self, tool: Tool):
        with self.session_maker() as session:
            if self.get_tool(tool_name=tool.name, user_id=tool.user_id) is not None:
                raise ValueError(f"Tool with name {tool.name} already exists")
            session.add(ToolModel(**vars(tool)))
            session.commit()

    @enforce_types
    def update_agent(self, agent: AgentState):
        with self.session_maker() as session:
            fields = vars(agent)
            if isinstance(agent.memory, Memory):  # TODO: this is nasty but this whole class will soon be removed so whatever
                fields["memory"] = agent.memory.to_dict()
            session.query(AgentModel).filter(AgentModel.id == agent.id).update(fields)
            session.commit()

    @enforce_types
    def update_user(self, user: User):
        with self.session_maker() as session:
            session.query(UserModel).filter(UserModel.id == user.id).update(vars(user))
            session.commit()

    @enforce_types
    def update_source(self, source: Source):
        with self.session_maker() as session:
            session.query(SourceModel).filter(SourceModel.id == source.id).update(vars(source))
            session.commit()

    @enforce_types
    def update_block(self, block: Block):
        with self.session_maker() as session:
            session.query(BlockModel).filter(BlockModel.id == block.id).update(vars(block))
            session.commit()

    @enforce_types
    def update_or_create_block(self, block: Block):
        with self.session_maker() as session:
            existing_block = session.query(BlockModel).filter(BlockModel.id == block.id).first()
            if existing_block:
                session.query(BlockModel).filter(BlockModel.id == block.id).update(vars(block))
            else:
                session.add(BlockModel(**vars(block)))
            session.commit()

    @enforce_types
    def update_tool(self, tool: Tool):
        with self.session_maker() as session:
            session.query(ToolModel).filter(ToolModel.id == tool.id).update(vars(tool))
            session.commit()

    @enforce_types
    def delete_tool(self, tool_id: str):
        with self.session_maker() as session:
            session.query(ToolModel).filter(ToolModel.id == tool_id).delete()
            session.commit()

    @enforce_types
    def delete_block(self, block_id: str):
        with self.session_maker() as session:
            session.query(BlockModel).filter(BlockModel.id == block_id).delete()
            session.commit()

    @enforce_types
    def delete_agent(self, agent_id: str):
        with self.session_maker() as session:

            # delete agents
            session.query(AgentModel).filter(AgentModel.id == agent_id).delete()

            # delete mappings
            session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.agent_id == agent_id).delete()

            session.commit()

    @enforce_types
    def delete_source(self, source_id: str):
        with self.session_maker() as session:
            # delete from sources table
            session.query(SourceModel).filter(SourceModel.id == source_id).delete()

            # delete any mappings
            session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.source_id == source_id).delete()

            session.commit()

    @enforce_types
    def delete_user(self, user_id: str):
        with self.session_maker() as session:
            # delete from users table
            session.query(UserModel).filter(UserModel.id == user_id).delete()

            # delete associated agents
            session.query(AgentModel).filter(AgentModel.user_id == user_id).delete()

            # delete associated sources
            session.query(SourceModel).filter(SourceModel.user_id == user_id).delete()

            # delete associated mappings
            session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.user_id == user_id).delete()

            session.commit()

    @enforce_types
    def delete_organization(self, org_id: str):
        with self.session_maker() as session:
            # delete from organizations table
            session.query(OrganizationModel).filter(OrganizationModel.id == org_id).delete()

            # TODO: delete associated data

            session.commit()

    @enforce_types
    # def list_tools(self, user_id: str) -> List[ToolModel]: # TODO: add when users can creat tools
    def list_tools(self, user_id: Optional[str] = None) -> List[ToolModel]:
        with self.session_maker() as session:
            results = session.query(ToolModel).filter(ToolModel.user_id == None).all()
            if user_id:
                results += session.query(ToolModel).filter(ToolModel.user_id == user_id).all()
            res = [r.to_record() for r in results]
            return res

    @enforce_types
    def list_agents(self, user_id: str) -> List[AgentState]:
        with self.session_maker() as session:
            results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def list_sources(self, user_id: str) -> List[Source]:
        with self.session_maker() as session:
            results = session.query(SourceModel).filter(SourceModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def get_agent(
        self, agent_id: Optional[str] = None, agent_name: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[AgentState]:
        with self.session_maker() as session:
            if agent_id:
                results = session.query(AgentModel).filter(AgentModel.id == agent_id).all()
            else:
                assert agent_name is not None and user_id is not None, "Must provide either agent_id or agent_name"
                results = session.query(AgentModel).filter(AgentModel.name == agent_name).filter(AgentModel.user_id == user_id).all()

            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"  # should only be one result
            return results[0].to_record()

    @enforce_types
    def get_user(self, user_id: str) -> Optional[User]:
        with self.session_maker() as session:
            results = session.query(UserModel).filter(UserModel.id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def get_organization(self, org_id: str) -> Optional[Organization]:
        with self.session_maker() as session:
            results = session.query(OrganizationModel).filter(OrganizationModel.id == org_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def list_organizations(self, cursor: Optional[str] = None, limit: Optional[int] = 50):
        with self.session_maker() as session:
            query = session.query(OrganizationModel).order_by(desc(OrganizationModel.id))
            if cursor:
                query = query.filter(OrganizationModel.id < cursor)
            results = query.limit(limit).all()
            if not results:
                return None, []
            organization_records = [r.to_record() for r in results]
            next_cursor = organization_records[-1].id
            assert isinstance(next_cursor, str)

            return next_cursor, organization_records

    @enforce_types
    def get_all_users(self, cursor: Optional[str] = None, limit: Optional[int] = 50):
        with self.session_maker() as session:
            query = session.query(UserModel).order_by(desc(UserModel.id))
            if cursor:
                query = query.filter(UserModel.id < cursor)
            results = query.limit(limit).all()
            if not results:
                return None, []
            user_records = [r.to_record() for r in results]
            next_cursor = user_records[-1].id
            assert isinstance(next_cursor, str)

            return next_cursor, user_records

    @enforce_types
    def get_source(
        self, source_id: Optional[str] = None, user_id: Optional[str] = None, source_name: Optional[str] = None
    ) -> Optional[Source]:
        with self.session_maker() as session:
            if source_id:
                results = session.query(SourceModel).filter(SourceModel.id == source_id).all()
            else:
                assert user_id is not None and source_name is not None
                results = session.query(SourceModel).filter(SourceModel.name == source_name).filter(SourceModel.user_id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def get_tool(
        self, tool_name: Optional[str] = None, tool_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[ToolModel]:
        with self.session_maker() as session:
            if tool_id:
                results = session.query(ToolModel).filter(ToolModel.id == tool_id).all()
            else:
                assert tool_name is not None
                results = session.query(ToolModel).filter(ToolModel.name == tool_name).filter(ToolModel.user_id == None).all()
                if user_id:
                    results += session.query(ToolModel).filter(ToolModel.name == tool_name).filter(ToolModel.user_id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def get_block(self, block_id: str) -> Optional[Block]:
        with self.session_maker() as session:
            results = session.query(BlockModel).filter(BlockModel.id == block_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def get_blocks(
        self,
        user_id: Optional[str],
        label: Optional[str] = None,
        template: Optional[bool] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Optional[List[Block]]:
        """List available blocks"""
        with self.session_maker() as session:
            query = session.query(BlockModel)

            if user_id:
                query = query.filter(BlockModel.user_id == user_id)

            if label:
                query = query.filter(BlockModel.label == label)

            if name:
                query = query.filter(BlockModel.name == name)

            if id:
                query = query.filter(BlockModel.id == id)

            if template:
                query = query.filter(BlockModel.template == template)

            results = query.all()

            if len(results) == 0:
                return None

            return [r.to_record() for r in results]

    # agent source metadata
    @enforce_types
    def attach_source(self, user_id: str, agent_id: str, source_id: str):
        with self.session_maker() as session:
            # TODO: remove this (is a hack)
            mapping_id = f"{user_id}-{agent_id}-{source_id}"
            session.add(AgentSourceMappingModel(id=mapping_id, user_id=user_id, agent_id=agent_id, source_id=source_id))
            session.commit()

    @enforce_types
    def list_attached_sources(self, agent_id: str) -> List[Source]:
        with self.session_maker() as session:
            results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.agent_id == agent_id).all()

            sources = []
            # make sure source exists
            for r in results:
                source = self.get_source(source_id=r.source_id)
                if source:
                    sources.append(source)
                else:
                    printd(f"Warning: source {r.source_id} does not exist but exists in mapping database. This should never happen.")
            return sources

    @enforce_types
    def list_attached_agents(self, source_id: str) -> List[str]:
        with self.session_maker() as session:
            results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.source_id == source_id).all()

            agent_ids = []
            # make sure agent exists
            for r in results:
                agent = self.get_agent(agent_id=r.agent_id)
                if agent:
                    agent_ids.append(r.agent_id)
                else:
                    printd(f"Warning: agent {r.agent_id} does not exist but exists in mapping database. This should never happen.")
            return agent_ids

    @enforce_types
    def detach_source(self, agent_id: str, source_id: str):
        with self.session_maker() as session:
            session.query(AgentSourceMappingModel).filter(
                AgentSourceMappingModel.agent_id == agent_id, AgentSourceMappingModel.source_id == source_id
            ).delete()
            session.commit()

    @enforce_types
    def create_job(self, job: Job):
        with self.session_maker() as session:
            session.add(JobModel(**vars(job)))
            session.commit()

    def delete_job(self, job_id: str):
        with self.session_maker() as session:
            session.query(JobModel).filter(JobModel.id == job_id).delete()
            session.commit()

    def get_job(self, job_id: str) -> Optional[Job]:
        with self.session_maker() as session:
            results = session.query(JobModel).filter(JobModel.id == job_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    def list_jobs(self, user_id: str) -> List[Job]:
        with self.session_maker() as session:
            results = session.query(JobModel).filter(JobModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    def update_job(self, job: Job) -> Job:
        with self.session_maker() as session:
            session.query(JobModel).filter(JobModel.id == job.id).update(vars(job))
            session.commit()
        return Job

    def update_job_status(self, job_id: str, status: JobStatus):
        with self.session_maker() as session:
            session.query(JobModel).filter(JobModel.id == job_id).update({"status": status})
            if status == JobStatus.COMPLETED:
                session.query(JobModel).filter(JobModel.id == job_id).update({"completed_at": get_utc_time()})
            session.commit()


class CommonVector(TypeDecorator):
    """Common type for representing vectors in SQLite"""

    impl = BINARY
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(BINARY())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        # Ensure value is a numpy array
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        # Serialize numpy array to bytes, then encode to base64 for universal compatibility
        return base64.b64encode(value.tobytes())

    def process_result_value(self, value, dialect):
        if not value:
            return value
        # Check database type and deserialize accordingly
        if dialect.name == "sqlite":
            # Decode from base64 and convert back to numpy array
            value = base64.b64decode(value)
        # For PostgreSQL, value is already in bytes
        return np.frombuffer(value, dtype=np.float32)


class MessageModel(Base):
    """Defines data model for storing Message objects"""

    __tablename__ = "messages"
    __table_args__ = {"extend_existing": True}

    # Assuming message_id is the primary key
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)

    # openai info
    role = Column(String, nullable=False)
    text = Column(String)  # optional: can be null if function call
    model = Column(String)  # optional: can be null if LLM backend doesn't require specifying
    name = Column(String)  # optional: multi-agent only

    # tool call request info
    # if role == "assistant", this MAY be specified
    # if role != "assistant", this must be null
    # TODO align with OpenAI spec of multiple tool calls
    # tool_calls = Column(ToolCallColumn)
    tool_calls = Column(ToolCallColumn)

    # tool call response info
    # if role == "tool", then this must be specified
    # if role != "tool", this must be null
    tool_call_id = Column(String)

    # Add a datetime column, with default value as the current time
    created_at = Column(DateTime(timezone=True))
    Index("message_idx_user", user_id, agent_id),

    def __repr__(self):
        return f"<Message(message_id='{self.id}', text='{self.text}')>"

    def to_record(self):
        # calls = (
        #    [ToolCall(id=tool_call["id"], function=ToolCallFunction(**tool_call["function"])) for tool_call in self.tool_calls]
        #    if self.tool_calls
        #    else None
        # )
        # if calls:
        #    assert isinstance(calls[0], ToolCall)
        if self.tool_calls and len(self.tool_calls) > 0:
            assert isinstance(self.tool_calls[0], ToolCall), type(self.tool_calls[0])
            for tool in self.tool_calls:
                assert isinstance(tool, ToolCall), type(tool)
        return Message(
            user_id=self.user_id,
            agent_id=self.agent_id,
            role=self.role,
            name=self.name,
            text=self.text,
            model=self.model,
            # tool_calls=[ToolCall(id=tool_call["id"], function=ToolCallFunction(**tool_call["function"])) for tool_call in self.tool_calls] if self.tool_calls else None,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            created_at=self.created_at,
            id=self.id,
        )


class PassageModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "passages"
    __table_args__ = {"extend_existing": True}

    # Assuming passage_id is the primary key
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    text = Column(String)
    doc_id = Column(String)
    agent_id = Column(String)
    source_id = Column(String)

    # vector storage
    if settings.letta_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
    elif config.archival_storage_type == "sqlite" or config.archival_storage_type == "chroma":
        embedding = Column(CommonVector)
    else:
        raise ValueError(f"Unsupported archival_storage_type: {config.archival_storage_type}")
    embedding_config = Column(EmbeddingConfigColumn)
    metadata_ = Column(MutableJson)

    # Add a datetime column, with default value as the current time
    created_at = Column(DateTime(timezone=True))

    Index("passage_idx_user", user_id, agent_id, doc_id),

    def __repr__(self):
        return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

    def to_record(self):
        return Passage(
            text=self.text,
            embedding=self.embedding,
            embedding_config=self.embedding_config,
            doc_id=self.doc_id,
            user_id=self.user_id,
            id=self.id,
            source_id=self.source_id,
            agent_id=self.agent_id,
            metadata_=self.metadata_,
            created_at=self.created_at,
        )


class SQLStorageConnector(StorageConnector):
    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        self.config = config

    def get_filters(self, filters: Optional[Dict] = {}):
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        all_filters = [getattr(self.db_model, key) == value for key, value in filter_conditions.items()]
        return all_filters

    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: Optional[int] = 1000, offset=0):
        filters = self.get_filters(filters)
        while True:
            # Retrieve a chunk of records with the given page_size
            with self.session_maker() as session:
                db_record_chunk = session.query(self.db_model).filter(*filters).offset(offset).limit(page_size).all()

            # If the chunk is empty, we've retrieved all records
            if not db_record_chunk:
                break

            # Yield a list of Record objects converted from the chunk
            yield [record.to_record() for record in db_record_chunk]

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
        with self.session_maker() as session:
            query = session.query(self.db_model).filter(*filters)
            # query = query.order_by(asc(self.db_model.id))

            # records are sorted by the order_by field first, and then by the ID if two fields are the same
            if reverse:
                query = query.order_by(desc(getattr(self.db_model, order_by)), asc(self.db_model.id))
            else:
                query = query.order_by(asc(getattr(self.db_model, order_by)), asc(self.db_model.id))

            # cursor logic: filter records based on before/after ID
            if after:
                after_value = getattr(self.get(id=after), order_by)
                sort_exp = getattr(self.db_model, order_by) > after_value
                query = query.filter(
                    or_(sort_exp, and_(getattr(self.db_model, order_by) == after_value, self.db_model.id > after))  # tiebreaker case
                )
            if before:
                before_value = getattr(self.get(id=before), order_by)
                sort_exp = getattr(self.db_model, order_by) < before_value
                query = query.filter(or_(sort_exp, and_(getattr(self.db_model, order_by) == before_value, self.db_model.id < before)))

            # get records
            db_record_chunk = query.limit(limit).all()
        if not db_record_chunk:
            return (None, [])
        records = [record.to_record() for record in db_record_chunk]
        next_cursor = db_record_chunk[-1].id
        assert isinstance(next_cursor, str)

        # return (cursor, list[records])
        return (next_cursor, records)

    def get_all(self, filters: Optional[Dict] = {}, limit=None):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            if limit:
                db_records = session.query(self.db_model).filter(*filters).limit(limit).all()
            else:
                db_records = session.query(self.db_model).filter(*filters).all()
        return [record.to_record() for record in db_records]

    def get(self, id: str):
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

    def insert(self, record):
        raise NotImplementedError

    def insert_many(self, records, show_progress=False):
        raise NotImplementedError

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}):
        raise NotImplementedError("Vector query not implemented for SQLStorageConnector")

    def save(self):
        return

    def list_data_sources(self):
        assert self.table_type == TableType.ARCHIVAL_MEMORY, f"list_data_sources only implemented for ARCHIVAL_MEMORY"
        with self.session_maker() as session:
            unique_data_sources = session.query(self.db_model.data_source).filter(*self.filters).distinct().all()
        return unique_data_sources

    def query_date(self, start_date, end_date, limit=None, offset=0):
        filters = self.get_filters({})
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(self.db_model.created_at >= start_date)
                .filter(self.db_model.created_at <= end_date)
                .filter(self.db_model.role != "system")
                .filter(self.db_model.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return [result.to_record() for result in results]

    def query_text(self, query, limit=None, offset=0):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        filters = self.get_filters({})
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(func.lower(self.db_model.text).contains(func.lower(query)))
                .filter(self.db_model.role != "system")
                .filter(self.db_model.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        # return [self.type(**vars(result)) for result in results]
        return [result.to_record() for result in results]

    # Should be used only in tests!
    def delete_table(self):
        close_all_sessions()
        with self.session_maker() as session:
            self.db_model.__table__.drop(session.bind)
            session.commit()

    def delete(self, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            session.query(self.db_model).filter(*filters).delete()
            session.commit()


class PostgresStorageConnector(SQLStorageConnector):
    """Storage via Postgres"""

    # TODO: this should probably eventually be moved into a parent DB class

    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        from pgvector.sqlalchemy import Vector

        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        # construct URI from enviornment variables
        if settings.pg_uri:
            self.uri = settings.pg_uri
        else:
            # use config URI
            # TODO: remove this eventually (config should NOT contain URI)
            if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
                self.uri = self.config.archival_storage_uri
                self.db_model = PassageModel
                if self.config.archival_storage_uri is None:
                    raise ValueError(f"Must specifiy archival_storage_uri in config {self.config.config_path}")
            elif table_type == TableType.RECALL_MEMORY:
                self.uri = self.config.recall_storage_uri
                self.db_model = MessageModel
                if self.config.recall_storage_uri is None:
                    raise ValueError(f"Must specifiy recall_storage_uri in config {self.config.config_path}")
            else:
                raise ValueError(f"Table type {table_type} not implemented")

        for c in self.db_model.__table__.columns:
            if c.name == "embedding":
                assert isinstance(c.type, Vector), f"Embedding column must be of type Vector, got {c.type}"

        from letta.server.server import db_context

        self.session_maker = db_context

        # TODO: move to DB init
        with self.session_maker() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)
        with self.session_maker() as session:
            results = session.scalars(
                select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)
            ).all()

        # Convert the results into Passage objects
        records = [result.to_record() for result in results]
        return records

    def insert_many(self, records, exists_ok=True, show_progress=False):
        # TODO: this is terrible, should eventually be done the same way for all types (migrate to SQLModel)
        if len(records) == 0:
            return

        added_ids = []  # avoid adding duplicates
        # NOTE: this has not great performance due to the excessive commits
        with self.session_maker() as session:
            iterable = tqdm(records) if show_progress else records
            for record in iterable:
                # db_record = self.db_model(**vars(record))

                if record.id in added_ids:
                    continue

                existing_record = session.query(self.db_model).filter_by(id=record.id).first()
                if existing_record:
                    if exists_ok:
                        fields = record.model_dump()
                        fields.pop("id")
                        session.query(self.db_model).filter(self.db_model.id == record.id).update(fields)
                        print(f"Updated record with id {record.id}")
                        session.commit()
                    else:
                        raise ValueError(f"Record with id {record.id} already exists.")

                else:
                    db_record = self.db_model(**record.dict())
                    session.add(db_record)
                    print(f"Added record with id {record.id}")
                    session.commit()

                added_ids.append(record.id)

    def insert(self, record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def update(self, record):
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

    def str_to_datetime(self, str_date: str) -> datetime:
        val = str_date.split("-")
        _datetime = datetime(int(val[0]), int(val[1]), int(val[2]))
        return _datetime

    def query_date(self, start_date, end_date, limit=None, offset=0):
        filters = self.get_filters({})
        _start_date = self.str_to_datetime(start_date) if isinstance(start_date, str) else start_date
        _end_date = self.str_to_datetime(end_date) if isinstance(end_date, str) else end_date
        with self.session_maker() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(self.db_model.created_at >= _start_date)
                .filter(self.db_model.created_at <= _end_date)
                .filter(self.db_model.role != "system")
                .filter(self.db_model.role != "tool")
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return [result.to_record() for result in results]


class SQLLiteStorageConnector(SQLStorageConnector):
    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)

        # get storage URI
        if table_type == TableType.ARCHIVAL_MEMORY or table_type == TableType.PASSAGES:
            raise ValueError(f"Table type {table_type} not implemented")
        elif table_type == TableType.RECALL_MEMORY:
            # TODO: eventually implement URI option
            self.path = self.config.recall_storage_path
            if self.path is None:
                raise ValueError(f"Must specifiy recall_storage_path in config {self.config.recall_storage_path}")
            self.db_model = MessageModel
        else:
            raise ValueError(f"Table type {table_type} not implemented")

        self.path = os.path.join(self.path, f"sqlite.db")

        from letta.server.server import db_context

        self.session_maker = db_context

        # import sqlite3

        # sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)
        # sqlite3.register_converter("UUID", lambda b: uuid.UUID(bytes_le=b))

    def insert_many(self, records, exists_ok=True, show_progress=False):
        # TODO: this is terrible, should eventually be done the same way for all types (migrate to SQLModel)
        if len(records) == 0:
            return

        added_ids = []  # avoid adding duplicates
        # NOTE: this has not great performance due to the excessive commits
        with self.session_maker() as session:
            iterable = tqdm(records) if show_progress else records
            for record in iterable:
                # db_record = self.db_model(**vars(record))

                if record.id in added_ids:
                    continue

                existing_record = session.query(self.db_model).filter_by(id=record.id).first()
                if existing_record:
                    if exists_ok:
                        fields = record.model_dump()
                        fields.pop("id")
                        session.query(self.db_model).filter(self.db_model.id == record.id).update(fields)
                        session.commit()
                    else:
                        raise ValueError(f"Record with id {record.id} already exists.")

                else:
                    db_record = self.db_model(**record.dict())
                    session.add(db_record)
                    session.commit()

                added_ids.append(record.id)

    def insert(self, record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def update(self, record):
        """
        Updates an existing record in the database with values from the provided record object.
        """
        if not record.id:
            raise ValueError("Record must have an id.")

        with self.session_maker() as session:
            # Fetch the existing record from the database
            db_record = session.query(self.db_model).filter_by(id=record.id).first()
            if not db_record:
                raise ValueError(f"Record with id {record.id} does not exist.")

            # Update the database record with values from the provided record object
            for column in self.db_model.__table__.columns:
                column_name = column.name
                if hasattr(record, column_name):
                    new_value = getattr(record, column_name)
                    setattr(db_record, column_name, new_value)

            # Commit the changes to the database
            session.commit()
