""" Metadata store for user/agent/data_source information"""

import os
import secrets
from typing import List, Optional, Union

from sqlalchemy import JSON, Column, DateTime, Index, String, TypeDecorator
from sqlalchemy.sql import func

from letta.config import LettaConfig
from letta.orm.base import Base
from letta.schemas.agent import PersistedAgentState
from letta.schemas.api_key import APIKey
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ToolRuleType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.tool_rule import ChildToolRule, InitToolRule, TerminalToolRule
from letta.schemas.user import User
from letta.services.per_agent_lock_manager import PerAgentLockManager
from letta.settings import settings
from letta.utils import enforce_types, printd


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


class ToolRulesColumn(TypeDecorator):
    """Custom type for storing a list of ToolRules as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        """Convert a list of ToolRules to JSON-serializable format."""
        if value:
            data = [rule.model_dump() for rule in value]
            for d in data:
                d["type"] = d["type"].value

            for d in data:
                assert not (d["type"] == "ToolRule" and "children" not in d), "ToolRule does not have children field"
            return data
        return value

    def process_result_value(self, value, dialect) -> List[Union[ChildToolRule, InitToolRule, TerminalToolRule]]:
        """Convert JSON back to a list of ToolRules."""
        if value:
            return [self.deserialize_tool_rule(rule_data) for rule_data in value]
        return value

    @staticmethod
    def deserialize_tool_rule(data: dict) -> Union[ChildToolRule, InitToolRule, TerminalToolRule]:
        """Deserialize a dictionary to the appropriate ToolRule subclass based on the 'type'."""
        rule_type = ToolRuleType(data.get("type"))  # Remove 'type' field if it exists since it is a class var
        if rule_type == ToolRuleType.run_first:
            return InitToolRule(**data)
        elif rule_type == ToolRuleType.exit_loop:
            return TerminalToolRule(**data)
        elif rule_type == ToolRuleType.constrain_child_tools:
            rule = ChildToolRule(**data)
            return rule
        else:
            raise ValueError(f"Unknown tool rule type: {rule_type}")


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
    system = Column(String)

    # configs
    agent_type = Column(String)
    llm_config = Column(LLMConfigColumn)
    embedding_config = Column(EmbeddingConfigColumn)

    # state
    metadata_ = Column(JSON)

    # tools
    tool_names = Column(JSON)
    tool_rules = Column(ToolRulesColumn)

    Index(__tablename__ + "_idx_user", user_id),

    def __repr__(self) -> str:
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> PersistedAgentState:
        agent_state = PersistedAgentState(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            created_at=self.created_at,
            description=self.description,
            message_ids=self.message_ids,
            system=self.system,
            tool_names=self.tool_names,
            tool_rules=self.tool_rules,
            agent_type=self.agent_type,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            metadata_=self.metadata_,
        )
        return agent_state


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
    def create_agent(self, agent: PersistedAgentState):
        # insert into agent table
        # make sure agent.name does not already exist for user user_id
        with self.session_maker() as session:
            if session.query(AgentModel).filter(AgentModel.name == agent.name).filter(AgentModel.user_id == agent.user_id).count() > 0:
                raise ValueError(f"Agent with name {agent.name} already exists")
            fields = vars(agent)
            # fields["memory"] = agent.memory.to_dict()
            # if "_internal_memory" in fields:
            #    del fields["_internal_memory"]
            # else:
            #    warnings.warn(f"Agent {agent.id} has no _internal_memory field")
            if "tags" in fields:
                del fields["tags"]
            # else:
            # warnings.warn(f"Agent {agent.id} has no tags field")
            session.add(AgentModel(**fields))
            session.commit()

    @enforce_types
    def update_agent(self, agent: PersistedAgentState):
        with self.session_maker() as session:
            fields = vars(agent)
            # if isinstance(agent.memory, Memory):  # TODO: this is nasty but this whole class will soon be removed so whatever
            #    fields["memory"] = agent.memory.to_dict()
            # if "_internal_memory" in fields:
            #    del fields["_internal_memory"]
            # else:
            #    warnings.warn(f"Agent {agent.id} has no _internal_memory field")
            if "tags" in fields:
                del fields["tags"]
            # else:
            # warnings.warn(f"Agent {agent.id} has no tags field")
            session.query(AgentModel).filter(AgentModel.id == agent.id).update(fields)
            session.commit()

    @enforce_types
    def delete_agent(self, agent_id: str, per_agent_lock_manager: PerAgentLockManager):
        # TODO: Remove this once Agent is on the ORM
        # TODO: To prevent unbounded growth
        per_agent_lock_manager.clear_lock(agent_id)

        with self.session_maker() as session:

            # delete agents
            session.query(AgentModel).filter(AgentModel.id == agent_id).delete()

            # delete mappings
            session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.agent_id == agent_id).delete()

            session.commit()

    @enforce_types
    def list_agents(self, user_id: str) -> List[PersistedAgentState]:
        with self.session_maker() as session:
            results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def get_agent(
        self, agent_id: Optional[str] = None, agent_name: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[PersistedAgentState]:
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

    # agent source metadata
    @enforce_types
    def attach_source(self, user_id: str, agent_id: str, source_id: str):
        with self.session_maker() as session:
            # TODO: remove this (is a hack)
            mapping_id = f"{user_id}-{agent_id}-{source_id}"
            session.add(AgentSourceMappingModel(id=mapping_id, user_id=user_id, agent_id=agent_id, source_id=source_id))
            session.commit()

    @enforce_types
    def list_attached_source_ids(self, agent_id: str) -> List[str]:
        with self.session_maker() as session:
            results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.agent_id == agent_id).all()
            return [r.source_id for r in results]

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
