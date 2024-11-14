from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field

from letta.schemas.letta_base import LettaBase, OrmMetadataBase


# Configs for different sandboxes
class LocalSandboxConfig(BaseModel):
    venv_name: str = Field("venv", description="Name of the virtual environment.")
    sandbox_dir: str = Field(..., description="Directory for the sandbox environment.")

    class Config:
        extra = "ignore"


class E2BConfig(BaseModel):
    timeout: int = Field(5 * 60, description="Time limit for the sandbox (in seconds).")
    template_id: Optional[str] = Field(None, description="The E2B template id (docker image).")

    class Config:
        extra = "ignore"


# Types
class SandboxType(str, Enum):
    E2B = "e2b"
    LOCAL_DIR = "local_dir"


# Sandbox Config
class SandboxConfigBase(OrmMetadataBase):
    __id_prefix__ = "sandbox"


class SandboxConfig(SandboxConfigBase):
    id: str = SandboxConfigBase.generate_id_field()
    type: SandboxType = Field(None, description="The type of sandbox.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the sandbox.")
    config: Dict = Field(default_factory=lambda: {}, description="The JSON configuration data.")

    def get_e2b_config(self) -> E2BConfig:
        return E2BConfig(**self.config)

    def get_local_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(**self.config)


class SandboxConfigUpdate(LettaBase):
    """Pydantic model for updating SandboxConfig fields."""

    config: Optional[Dict] = Field(None, description="The JSON configuration data for the sandbox.")

    class Config:
        extra = "ignore"


# Environment Variable
class SandboxEnvironmentVariableBase(OrmMetadataBase):
    __id_prefix__ = "sandbox-env"


class SandboxEnvironmentVariable(SandboxEnvironmentVariableBase):
    id: str = SandboxEnvironmentVariableBase.generate_id_field()
    key: str = Field(..., description="The name of the environment variable.")
    value: str = Field(..., description="The value of the environment variable.")
    description: Optional[str] = Field(None, description="An optional description of the environment variable.")
    organization_id: Optional[str] = Field(None, description="The ID of the organization this environment variable belongs to.")


class SandboxEnvVarUpdate(LettaBase):
    """Pydantic model for updating SandboxEnvironmentVariable fields."""

    key: Optional[str] = Field(None, description="The name of the environment variable.")
    value: Optional[str] = Field(None, description="The value of the environment variable.")
    description: Optional[str] = Field(None, description="An optional description of the environment variable.")

    class Config:
        extra = "ignore"
