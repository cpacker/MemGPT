from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# this is the driver we use
POSTGRES_SCHEME = "postgresql+pg8000"


class BackendConfiguration(BaseModel):
    name: Literal["postgres", "sqlite_chroma"]
    database_uri: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="memgpt_")
    memgpt_dir: Optional[Path] = Field(Path.home() / ".memgpt", env="MEMGPT_DIR")
    debug: Optional[bool] = False
    server_pass: Optional[str] = None
    cors_origins: Optional[list] = ["http://memgpt.localhost", "http://localhost:8283", "http://localhost:8083"]

    # backend database settings
    pg_uri: Optional[str] = Field(None, description="if set backend will use postgresql. Othewise default to sqlite.")

    @property
    def backend(self) -> BackendConfiguration:
        """Return an adjusted BackendConfiguration.
        Note: defaults to sqlite_chroma if pg_uri is not set.
        """
        if self.pg_uri:
            return BackendConfiguration(name="postgres", database_uri=self._correct_pg_uri(self.pg_uri))
        return BackendConfiguration(name="sqlite_chroma", database_uri=f"sqlite:///{self.memgpt_dir}/memgpt.db")

    @classmethod
    def _correct_pg_uri(cls, uri: str) -> str:
        """It is awkward to have users set a scheme for the uri (because why should they know anything about what drivers we use?)
        So here we check (and correct) the provided uri to use the scheme we implement.
        """
        url_parts = list(urlsplit(uri))
        SCHEME = 0
        url_parts[SCHEME] = POSTGRES_SCHEME
        return urlunsplit(url_parts)

    # configurations
    config_path: Optional[Path] = Path("~/.memgpt/config").expanduser()

    # application default starter settings
    persona: Optional[str] = "sam_pov"
    human: Optional[str] = "basic"
    preset: Optional[str] = "memgpt_chat"

    # TODO: extract to vendor plugin
    openai_api_key: Optional[str] = None


class TestSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="memgpt_test_")

    memgpt_dir: Optional[Path] = Field(Path.home() / ".memgpt/test", env="MEMGPT_TEST_DIR")


class TestSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="memgpt_test_")

    memgpt_dir: Optional[Path] = Field(Path.home() / ".memgpt/test", env="MEMGPT_TEST_DIR")


# singleton
settings = Settings()
test_settings = TestSettings()
