from enum import Enum
from typing import Dict, Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase, OrmMetadataBase


# Types
class SandboxType(str, Enum):
    E2B = "e2b"
    LOCAL = "local"


# Sandbox Config
class SandboxConfigBase(OrmMetadataBase):
    __id_prefix__ = "sandbox"


class SandboxConfig(SandboxConfigBase):
    id: str = SandboxConfigBase.generate_id_field()
    type: SandboxType = Field(None, description="The type of sandbox.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the sandbox.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the sandbox.")
    config: Dict = Field(..., description="The JSON configuration data.")


class SandboxConfigUpdate(LettaBase):
    """Pydantic model for updating SandboxConfig fields."""

    metadata_: Optional[Dict] = Field(None, description="Metadata associated with the sandbox.")
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
