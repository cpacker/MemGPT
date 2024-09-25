import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.utils import printd


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="letta_")

    letta_dir: Optional[Path] = Field(Path.home() / ".letta", env="LETTA_DIR")
    debug: Optional[bool] = False
    cors_origins: Optional[list] = ["http://letta.localhost", "http://localhost:8283", "http://localhost:8083"]

    # database configuration
    pg_db: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_uri: Optional[str] = None  # option to specifiy full uri

    # llm configuration
    llm_endpoint: Optional[str] = None
    llm_endpoint_type: Optional[str] = None
    llm_model: Optional[str] = None
    llm_context_window: Optional[int] = None

    # embedding configuration
    embedding_endpoint: Optional[str] = None
    embedding_endpoint_type: Optional[str] = None
    embedding_dim: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_chunk_size: int = 300

    @property
    def llm_config(self):

        # try to get LLM config from settings
        if self.llm_endpoint and self.llm_endpoint_type and self.llm_model and self.llm_context_window:
            return LLMConfig(
                model=self.llm_model,
                model_endpoint_type=self.llm_endpoint_type,
                model_endpoint=self.llm_endpoint,
                model_wrapper=None,
                context_window=self.llm_context_window,
            )
        else:
            if not self.llm_endpoint:
                printd(f"No LETTA_LLM_ENDPOINT provided")
            if not self.llm_endpoint_type:
                printd(f"No LETTA_LLM_ENDPOINT_TYPE provided")
            if not self.llm_model:
                printd(f"No LETTA_LLM_MODEL provided")
            if not self.llm_context_window:
                printd(f"No LETTA_LLM_CONTEX_WINDOW provided")

        # quickstart options
        if self.llm_model:
            try:
                return LLMConfig.default_config(self.llm_model)
            except ValueError:
                pass

        # try to read from config file (last resort)
        from letta.config import LettaConfig

        if LettaConfig.exists():
            config = LettaConfig.load()
            llm_config = LLMConfig(
                model=config.default_llm_config.model,
                model_endpoint_type=config.default_llm_config.model_endpoint_type,
                model_endpoint=config.default_llm_config.model_endpoint,
                model_wrapper=config.default_llm_config.model_wrapper,
                context_window=config.default_llm_config.context_window,
            )
            return llm_config

        # check OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            return LLMConfig.default_config(self.llm_model if self.llm_model else "gpt-4")

        return LLMConfig.default_config("letta")

    @property
    def embedding_config(self):

        # try to get LLM config from settings
        if self.embedding_endpoint and self.embedding_endpoint_type and self.embedding_model and self.embedding_dim:
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type=self.embedding_endpoint_type,
                embedding_endpoint=self.embedding_endpoint,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=self.embedding_chunk_size,
            )
        else:
            if not self.embedding_endpoint:
                printd(f"No LETTA_EMBEDDING_ENDPOINT provided")
            if not self.embedding_endpoint_type:
                printd(f"No LETTA_EMBEDDING_ENDPOINT_TYPE provided")
            if not self.embedding_model:
                printd(f"No LETTA_EMBEDDING_MODEL provided")
            if not self.embedding_dim:
                printd(f"No LETTA_EMBEDDING_DIM provided")

        # TODO
        ## quickstart options
        # if self.embedding_model:
        #    try:
        #        return EmbeddingConfig.default_config(self.embedding_model)
        #    except ValueError as e:
        #        pass

        # try to read from config file (last resort)
        from letta.config import LettaConfig

        if LettaConfig.exists():
            config = LettaConfig.load()
            return EmbeddingConfig(
                embedding_model=config.default_embedding_config.embedding_model,
                embedding_endpoint_type=config.default_embedding_config.embedding_endpoint_type,
                embedding_endpoint=config.default_embedding_config.embedding_endpoint,
                embedding_dim=config.default_embedding_config.embedding_dim,
                embedding_chunk_size=config.default_embedding_config.embedding_chunk_size,
            )

        if os.getenv("OPENAI_API_KEY"):
            return EmbeddingConfig.default_config(self.embedding_model if self.embedding_model else "text-embedding-ada-002")

        return EmbeddingConfig.default_config("letta")

    @property
    def letta_pg_uri(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return f"postgresql+pg8000://letta:letta@localhost:5432/letta"

    # add this property to avoid being returned the default
    # reference: https://github.com/cpacker/Letta/issues/1362
    @property
    def letta_pg_uri_no_default(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return None


class TestSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="letta_test_")

    letta_dir: Optional[Path] = Field(Path.home() / ".letta/test", env="LETTA_TEST_DIR")


# singleton
settings = Settings(_env_parse_none_str="None")
test_settings = TestSettings()
