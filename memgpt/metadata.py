""" Metadata store for user/agent/data_source information"""

import os
import inspect as python_inspect
import uuid
import secrets
from typing import Optional, List

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_PRESET, LLM_MAX_TOKENS
from memgpt.utils import get_local_time, enforce_types
from memgpt.data_types import AgentState, Source, User, LLMConfig, EmbeddingConfig, Token, Preset
from memgpt.config import MemGPTConfig
from memgpt.functions.functions import load_all_function_sets

from memgpt.models.pydantic_models import PersonaModel, HumanModel, ToolModel

from sqlalchemy import create_engine, Column, String, BIGINT, select, inspect, text, JSON, BLOB, BINARY, ARRAY, Boolean
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, mapped_column, declarative_base
from sqlalchemy.orm.session import close_all_sessions
from sqlalchemy.sql import func
from sqlalchemy import Column, BIGINT, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy_json import mutable_json_type, MutableJson
from sqlalchemy import TypeDecorator, CHAR
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
        # print("GET VALUE", value)
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
    # name = Column(String, nullable=False)
    default_agent = Column(String)

    policies_accepted = Column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<User(id='{self.id}')>"

    def to_record(self) -> User:
        return User(
            id=self.id,
            # name=self.name
            default_agent=self.default_agent,
            policies_accepted=self.policies_accepted,
        )


class TokenModel(Base):
    """Data model for authentication tokens. One-to-many relationship with UserModel (1 User - N tokens)."""

    __tablename__ = "tokens"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    # each api key is tied to a user account (that it validates access for)
    user_id = Column(CommonUUID, nullable=False)
    # the api key
    token = Column(String, nullable=False)
    # extra (optional) metadata
    name = Column(String)

    def __repr__(self) -> str:
        return f"<Token(id='{self.id}', token='{self.token}', name='{self.name}')>"

    def to_record(self) -> User:
        return Token(
            id=self.id,
            user_id=self.user_id,
            token=self.token,
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
    embedding_dim = Column(BIGINT)
    embedding_model = Column(String)

    # TODO: add num passages

    def __repr__(self) -> str:
        return f"<Source(passage_id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Source:
        return Source(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            created_at=self.created_at,
            embedding_dim=self.embedding_dim,
            embedding_model=self.embedding_model,
        )


class AgentSourceMappingModel(Base):
    """Stores mapping between agent -> source"""

    __tablename__ = "agent_source_mapping"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    agent_id = Column(CommonUUID, nullable=False)
    source_id = Column(CommonUUID, nullable=False)

    def __repr__(self) -> str:
        return f"<AgentSourceMapping(user_id='{self.user_id}', agent_id='{self.agent_id}', source_id='{self.source_id}')>"


class PresetSourceMapping(Base):
    __tablename__ = "preset_source_mapping"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    preset_id = Column(CommonUUID, nullable=False)
    source_id = Column(CommonUUID, nullable=False)

    def __repr__(self) -> str:
        return f"<PresetSourceMapping(user_id='{self.user_id}', preset_id='{self.preset_id}', source_id='{self.source_id}')>"


# class PresetFunctionMapping(Base):
#    __tablename__ = "preset_function_mapping"
#
#    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
#    user_id = Column(CommonUUID, nullable=False)
#    preset_id = Column(CommonUUID, nullable=False)
#    #function_id = Column(CommonUUID, nullable=False)
#    function = Column(String, nullable=False) # TODO: convert to ID eventually
#
#    def __repr__(self) -> str:
#        return f"<PresetFunctionMapping(user_id='{self.user_id}', preset_id='{self.preset_id}', function_id='{self.function_id}')>"


class PresetModel(Base):
    """Defines data model for storing Preset objects"""

    __tablename__ = "presets"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    system = Column(String)
    human = Column(String)
    human_name = Column(String, nullable=False)
    persona = Column(String)
    persona_name = Column(String, nullable=False)
    preset = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    functions_schema = Column(JSON)

    def __repr__(self) -> str:
        return f"<Preset(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Preset:
        return Preset(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            description=self.description,
            system=self.system,
            human=self.human,
            persona=self.persona,
            human_name=self.human_name,
            persona_name=self.persona_name,
            preset=self.preset,
            created_at=self.created_at,
            functions_schema=self.functions_schema,
        )


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

        # Ensure valid URI
        if not self.uri:
            raise ValueError("Database URI is not provided or is invalid.")

        # Check if tables need to be created
        self.engine = create_engine(self.uri)
        Base.metadata.create_all(
            self.engine,
            tables=[
                UserModel.__table__,
                AgentModel.__table__,
                SourceModel.__table__,
                AgentSourceMappingModel.__table__,
                TokenModel.__table__,
                PresetModel.__table__,
                PresetSourceMapping.__table__,
                HumanModel.__table__,
                PersonaModel.__table__,
            ],
        )
        self.session_maker = sessionmaker(bind=self.engine)

    @enforce_types
    def create_api_key(self, user_id: uuid.UUID, name: Optional[str] = None) -> Token:
        """Create an API key for a user"""
        new_api_key = generate_api_key()
        with self.session_maker() as session:
            if session.query(TokenModel).filter(TokenModel.token == new_api_key).count() > 0:
                # NOTE duplicate API keys / tokens should never happen, but if it does don't allow it
                raise ValueError(f"Token {new_api_key} already exists")
            # TODO store the API keys as hashed
            token = Token(user_id=user_id, token=new_api_key, name=name)
            session.add(TokenModel(**vars(token)))
            session.commit()
        return self.get_api_key(api_key=new_api_key)

    @enforce_types
    def delete_api_key(self, api_key: str):
        """Delete an API key from the database"""
        with self.session_maker() as session:
            session.query(TokenModel).filter(TokenModel.token == api_key).delete()
            session.commit()

    @enforce_types
    def get_api_key(self, api_key: str) -> Optional[Token]:
        with self.session_maker() as session:
            results = session.query(TokenModel).filter(TokenModel.token == api_key).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"  # should only be one result
            return results[0].to_record()

    @enforce_types
    def get_all_api_keys_for_user(self, user_id: uuid.UUID) -> List[Token]:
        with self.session_maker() as session:
            results = session.query(TokenModel).filter(TokenModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def get_user_from_api_key(self, api_key: str) -> Optional[User]:
        """Get the user associated with a given API key"""
        token = self.get_api_key(api_key=api_key)
        if token is None:
            raise ValueError(f"Token {api_key} does not exist")
        else:
            return self.get_user(user_id=token.user_id)

    @enforce_types
    def create_agent(self, agent: AgentState):
        # insert into agent table
        # make sure agent.name does not already exist for user user_id
        with self.session_maker() as session:
            if session.query(AgentModel).filter(AgentModel.name == agent.name).filter(AgentModel.user_id == agent.user_id).count() > 0:
                raise ValueError(f"Agent with name {agent.name} already exists")
            session.add(AgentModel(**vars(agent)))
            session.commit()

    @enforce_types
    def create_source(self, source: Source, exists_ok=False):
        # make sure source.name does not already exist for user
        with self.session_maker() as session:
            if session.query(SourceModel).filter(SourceModel.name == source.name).filter(SourceModel.user_id == source.user_id).count() > 0:
                if not exists_ok:
                    raise ValueError(f"Source with name {source.name} already exists for user {source.user_id}")
                else:
                    session.update(SourceModel(**vars(source)))
            else:
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
    def create_preset(self, preset: Preset):
        with self.session_maker() as session:
            if session.query(PresetModel).filter(PresetModel.id == preset.id).count() > 0:
                raise ValueError(f"User with id {preset.id} already exists")
            session.add(PresetModel(**vars(preset)))
            session.commit()

    @enforce_types
    def get_preset(
        self, preset_id: Optional[uuid.UUID] = None, name: Optional[str] = None, user_id: Optional[uuid.UUID] = None
    ) -> Optional[Preset]:
        with self.session_maker() as session:
            if preset_id:
                results = session.query(PresetModel).filter(PresetModel.id == preset_id).all()
            elif name and user_id:
                results = session.query(PresetModel).filter(PresetModel.name == name).filter(PresetModel.user_id == user_id).all()
            else:
                raise ValueError("Must provide either preset_id or (preset_name and user_id)")
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    # @enforce_types
    # def set_preset_functions(self, preset_id: uuid.UUID, functions: List[str]):
    #    preset = self.get_preset(preset_id)
    #    if preset is None:
    #        raise ValueError(f"Preset with id {preset_id} does not exist")
    #    user_id = preset.user_id
    #    with self.session_maker() as session:
    #        for function in functions:
    #            session.add(PresetFunctionMapping(user_id=user_id, preset_id=preset_id, function=function))
    #        session.commit()

    @enforce_types
    def set_preset_sources(self, preset_id: uuid.UUID, sources: List[uuid.UUID]):
        preset = self.get_preset(preset_id)
        if preset is None:
            raise ValueError(f"Preset with id {preset_id} does not exist")
        user_id = preset.user_id
        with self.session_maker() as session:
            for source_id in sources:
                session.add(PresetSourceMapping(user_id=user_id, preset_id=preset_id, source_id=source_id))
            session.commit()

    # @enforce_types
    # def get_preset_functions(self, preset_id: uuid.UUID) -> List[str]:
    #    with self.session_maker() as session:
    #        results = session.query(PresetFunctionMapping).filter(PresetFunctionMapping.preset_id == preset_id).all()
    #        return [r.function for r in results]

    @enforce_types
    def get_preset_sources(self, preset_id: uuid.UUID) -> List[uuid.UUID]:
        with self.session_maker() as session:
            results = session.query(PresetSourceMapping).filter(PresetSourceMapping.preset_id == preset_id).all()
            return [r.source_id for r in results]

    @enforce_types
    def update_agent(self, agent: AgentState):
        with self.session_maker() as session:
            session.query(AgentModel).filter(AgentModel.id == agent.id).update(vars(agent))
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
    def delete_agent(self, agent_id: uuid.UUID):
        with self.session_maker() as session:
            session.query(AgentModel).filter(AgentModel.id == agent_id).delete()
            session.commit()

    @enforce_types
    def delete_source(self, source_id: uuid.UUID):
        with self.session_maker() as session:
            # delete from sources table
            session.query(SourceModel).filter(SourceModel.id == source_id).delete()

            # delete any mappings
            session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.source_id == source_id).delete()

            session.commit()

    @enforce_types
    def delete_user(self, user_id: uuid.UUID):
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
    def list_presets(self, user_id: uuid.UUID) -> List[Preset]:
        with self.session_maker() as session:
            results = session.query(PresetModel).filter(PresetModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def list_tools(self, user_id: uuid.UUID) -> List[ToolModel]:
        with self.session_maker() as session:
            available_functions = load_all_function_sets()
            print(available_functions)
            results = [
                ToolModel(
                    name=k,
                    json_schema=v["json_schema"],
                    tags=v["tags"],
                    source_type="python",
                    source_code=python_inspect.getsource(v["python_function"]),
                )
                for k, v in available_functions.items()
            ]
            print(results)
            return results
            # results = session.query(PresetModel).filter(PresetModel.user_id == user_id).all()
            # return [r.to_record() for r in results]

    @enforce_types
    def list_agents(self, user_id: uuid.UUID) -> List[AgentState]:
        with self.session_maker() as session:
            results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def list_sources(self, user_id: uuid.UUID) -> List[Source]:
        with self.session_maker() as session:
            results = session.query(SourceModel).filter(SourceModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def get_agent(
        self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None, user_id: Optional[uuid.UUID] = None
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
    def get_user(self, user_id: uuid.UUID) -> Optional[User]:
        with self.session_maker() as session:
            results = session.query(UserModel).filter(UserModel.id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def get_all_users(self) -> List[User]:
        # TODO make paginated
        with self.session_maker() as session:
            results = session.query(UserModel).all()
            return [r.to_record() for r in results]

    @enforce_types
    def get_source(
        self, source_id: Optional[uuid.UUID] = None, user_id: Optional[uuid.UUID] = None, source_name: Optional[str] = None
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

    # agent source metadata
    @enforce_types
    def attach_source(self, user_id: uuid.UUID, agent_id: uuid.UUID, source_id: uuid.UUID):
        with self.session_maker() as session:
            session.add(AgentSourceMappingModel(user_id=user_id, agent_id=agent_id, source_id=source_id))
            session.commit()

    @enforce_types
    def list_attached_sources(self, agent_id: uuid.UUID) -> List[uuid.UUID]:
        with self.session_maker() as session:
            results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.agent_id == agent_id).all()
            return [r.source_id for r in results]

    @enforce_types
    def list_attached_agents(self, source_id: uuid.UUID) -> List[uuid.UUID]:
        with self.session_maker() as session:
            results = session.query(AgentSourceMappingModel).filter(AgentSourceMappingModel.source_id == source_id).all()
            return [r.agent_id for r in results]

    @enforce_types
    def detach_source(self, agent_id: uuid.UUID, source_id: uuid.UUID):
        with self.session_maker() as session:
            session.query(AgentSourceMappingModel).filter(
                AgentSourceMappingModel.agent_id == agent_id, AgentSourceMappingModel.source_id == source_id
            ).delete()
            session.commit()

    @enforce_types
    def add_human(self, human: HumanModel):
        with self.session_maker() as session:
            session.add(human)
            session.commit()

    @enforce_types
    def add_persona(self, persona: PersonaModel):
        with self.session_maker() as session:
            session.add(persona)
            session.commit()

    @enforce_types
    def add_preset(self, preset: PresetModel):
        with self.session_maker() as session:
            session.add(preset)
            session.commit()

    @enforce_types
    def get_human(self, name: str, user_id: uuid.UUID) -> str:
        with self.session_maker() as session:
            results = session.query(HumanModel).filter(HumanModel.name == name).filter(HumanModel.user_id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0]

    @enforce_types
    def get_persona(self, name: str, user_id: uuid.UUID) -> str:
        with self.session_maker() as session:
            results = session.query(PersonaModel).filter(PersonaModel.name == name).filter(PersonaModel.user_id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0]

    @enforce_types
    def list_personas(self, user_id: uuid.UUID) -> List[PersonaModel]:
        with self.session_maker() as session:
            results = session.query(PersonaModel).filter(PersonaModel.user_id == user_id).all()
            return results

    @enforce_types
    def list_humans(self, user_id: uuid.UUID) -> List[HumanModel]:
        with self.session_maker() as session:
            # if user_id matches provided user_id or if user_id is None
            results = session.query(HumanModel).filter(HumanModel.user_id == user_id).all()
            return results

    @enforce_types
    def list_presets(self, user_id: uuid.UUID) -> List[PresetModel]:
        with self.session_maker() as session:
            results = session.query(PresetModel).filter(PresetModel.user_id == user_id).all()
            return results

    @enforce_types
    def delete_human(self, name: str, user_id: uuid.UUID):
        with self.session_maker() as session:
            session.query(HumanModel).filter(HumanModel.name == name).filter(HumanModel.user_id == user_id).delete()
            session.commit()

    @enforce_types
    def delete_persona(self, name: str, user_id: uuid.UUID):
        with self.session_maker() as session:
            session.query(PersonaModel).filter(PersonaModel.name == name).filter(PersonaModel.user_id == user_id).delete()
            session.commit()

    @enforce_types
    def delete_preset(self, name: str, user_id: uuid.UUID):
        with self.session_maker() as session:
            session.query(PresetModel).filter(PresetModel.name == name).filter(PresetModel.user_id == user_id).delete()
            session.commit()
