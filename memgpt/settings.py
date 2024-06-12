from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="memgpt_")

    server_pass: Optional[str] = None
    pg_db: Optional[str] = "memgpt"
    pg_user: Optional[str] = "memgpt"
    pg_password: Optional[str] = "memgpt"
    pg_host: Optional[str] = "localhost"
    pg_port: Optional[int] = 5432
    cors_origins: Optional[list] = ["http://memgpt.localhost", "http://localhost:8283", "http://localhost:8083"]
    _pg_uri: Optional[str] = None  # calculated to specify full uri
    # configurations
    config_path: Optional[Path] = Path("~/.memgpt/config").expanduser()

    @property
    def pg_uri(self) -> str:
        if self._pg_uri:
            return self._pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return None

    @pg_uri.setter
    def pg_uri(self, value: str):
        self._pg_uri = value




# singleton
settings = Settings()
