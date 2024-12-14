""" Metadata store for user/agent/data_source information"""

from sqlalchemy import JSON, TypeDecorator

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig


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
