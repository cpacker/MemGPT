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
    pg_uri: Optional[str] = None  # option to specifiy full uri
    cors_origins: Optional[list] = ["http://memgpt.localhost", "http://localhost:8283", "http://localhost:8083"]

    @property
    def memgpt_pg_uri(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return None


# singleton
settings = Settings()
