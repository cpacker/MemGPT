from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from letta.local_llm.constants import DEFAULT_WRAPPER_NAME


class ToolSettings(BaseSettings):
    composio_api_key: Optional[str] = None


class ModelSettings(BaseSettings):

    # env_prefix='my_prefix_'

    # when we use /completions APIs (instead of /chat/completions), we need to specify a model wrapper
    # the "model wrapper" is responsible for prompt formatting and function calling parsing
    default_prompt_formatter: str = DEFAULT_WRAPPER_NAME

    # openai
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"

    # groq
    groq_api_key: Optional[str] = None

    # anthropic
    anthropic_api_key: Optional[str] = None

    # ollama
    ollama_base_url: Optional[str] = None

    # azure
    azure_api_key: Optional[str] = None
    azure_base_url: Optional[str] = None
    # We provide a default here, since usually people will want to be on the latest API version.
    azure_api_version: Optional[str] = (
        "2024-09-01-preview"  # https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
    )

    # google ai
    gemini_api_key: Optional[str] = None

    # vLLM
    vllm_api_base: Optional[str] = None

    # openllm
    openllm_auth_type: Optional[str] = None
    openllm_api_key: Optional[str] = None


cors_origins = ["http://letta.localhost", "http://localhost:8283", "http://localhost:8083", "http://localhost:3000"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="letta_")

    letta_dir: Optional[Path] = Field(Path.home() / ".letta", env="LETTA_DIR")
    debug: Optional[bool] = False
    cors_origins: Optional[list] = cors_origins

    # database configuration
    pg_db: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_uri: Optional[str] = None  # option to specifiy full uri

    # tools configuration
    load_default_external_tools: Optional[bool] = None

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
model_settings = ModelSettings()
tool_settings = ToolSettings()
