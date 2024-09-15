from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="letta_")

    letta_dir: Optional[Path] = Field(Path.home() / ".letta", env="LETTA_DIR")
    debug: Optional[bool] = False
    server_pass: Optional[str] = None
    pg_db: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_uri: Optional[str] = None  # option to specifiy full uri
    cors_origins: Optional[list] = ["http://letta.localhost", "http://localhost:8283", "http://localhost:8083"]

    # agent configuration defaults
    default_preset: Optional[str] = "memgpt_chat"

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

    letta_dir: Optional[Path] = Field(Path.home() / ".letta/test", env="MEMGPT_TEST_DIR")


# singleton
settings = Settings()
test_settings = TestSettings()
