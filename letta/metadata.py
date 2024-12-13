""" Metadata store for user/agent/data_source information"""

import os
import secrets
from typing import List, Optional

from sqlalchemy import JSON, Column, Index, String, TypeDecorator

from letta.config import LettaConfig
from letta.orm.base import Base
from letta.schemas.api_key import APIKey
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.user import User
from letta.settings import settings
from letta.utils import enforce_types


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
