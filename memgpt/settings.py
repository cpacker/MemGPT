from pathlib import Path
from typing import Optional
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class StorageType(str, Enum):
    sqlite = "sqlite"
    postgres = "postgres"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="memgpt_")
    storage_type: Optional[StorageType] = Field(description="What is the RDBMS type associated with the database url?", default=StorageType.sqlite)
    memgpt_dir: Optional[Path] = Field(Path.home() / ".memgpt", env="MEMGPT_DIR")
    debug: Optional[bool] = False
    server_pass: Optional[str] = None
    cors_origins: Optional[list] = ["http://memgpt.localhost", "http://localhost:8283", "http://localhost:8083"]
    pg_db: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    _pg_uri: Optional[str] = None  # calculated to specify full uri
    # configurations
    config_path: Optional[Path] = Path("~/.memgpt/config").expanduser()

    # application default starter settings
    persona: Optional[str] = "sam_pov"
    human: Optional[str] = "basic"
    preset: Optional[str] = "memgpt_chat"

    # TODO: extract to vendor plugin
    openai_api_key: Optional[str] = None

    @property
    def database_url(self) -> str:
        if self.storage_type == StorageType.sqlite:
            return f"sqlite:///{self.memgpt_dir}/memgpt.db"
        return self.pg_uri

    @property
    def pg_uri(self) -> str:
        if self._pg_uri:
            return self._pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return f"postgresql+pg8000://memgpt:memgpt@localhost:5432/memgpt"

    # add this property to avoid being returned the default
    # reference: https://github.com/cpacker/MemGPT/issues/1362
    @property
    def memgpt_pg_uri_no_default(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return None

    @pg_uri.setter
    def pg_uri(self, value: str):
        self._pg_uri = value




# singleton
settings = Settings()
