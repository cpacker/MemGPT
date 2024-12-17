import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from letta.schemas.agent import AgentState
from letta.schemas.letta_base import LettaBase, OrmMetadataBase
from letta.settings import tool_settings


# Sandbox Config
class SandboxType(str, Enum):
    E2B = "e2b"
    LOCAL = "local"


class SandboxRunResult(BaseModel):
    func_return: Optional[Any] = Field(None, description="The function return object")
    agent_state: Optional[AgentState] = Field(None, description="The agent state")
    stdout: Optional[List[str]] = Field(None, description="Captured stdout (e.g. prints, logs) from the function invocation")
    stderr: Optional[List[str]] = Field(None, description="Captured stderr from the function invocation")
    status: Literal["success", "error"] = Field(..., description="The status of the tool execution and return object")
    sandbox_config_fingerprint: str = Field(None, description="The fingerprint of the config for the sandbox")


class LocalSandboxConfig(BaseModel):
    sandbox_dir: str = Field(..., description="Directory for the sandbox environment.")
    use_venv: bool = Field(False, description="Whether or not to use the venv, or run directly in the same run loop.")
    venv_name: str = Field(
        "venv",
        description="The name for the venv in the sandbox directory. We first search for an existing venv with this name, otherwise, we make it from the requirements.txt.",
    )

    @property
    def type(self) -> "SandboxType":
        return SandboxType.LOCAL


class E2BSandboxConfig(BaseModel):
    timeout: int = Field(5 * 60, description="Time limit for the sandbox (in seconds).")
    template: Optional[str] = Field(None, description="The E2B template id (docker image).")
    pip_requirements: Optional[List[str]] = Field(None, description="A list of pip packages to install on the E2B Sandbox")

    @property
    def type(self) -> "SandboxType":
        return SandboxType.E2B

    @model_validator(mode="before")
    @classmethod
    def set_default_template(cls, data: dict):
        """
        Assign a default template value if the template field is not provided.
        """
        if data.get("template") is None:
            data["template"] = tool_settings.e2b_sandbox_template_id
        return data


class SandboxConfigBase(OrmMetadataBase):
    __id_prefix__ = "sandbox"


class SandboxConfig(SandboxConfigBase):
    id: str = SandboxConfigBase.generate_id_field()
    type: SandboxType = Field(None, description="The type of sandbox.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the sandbox.")
    config: Dict = Field(default_factory=lambda: {}, description="The JSON sandbox settings data.")

    def get_e2b_config(self) -> E2BSandboxConfig:
        return E2BSandboxConfig(**self.config)

    def get_local_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(**self.config)

    def fingerprint(self) -> str:
        # Only take into account type, org_id, and the config items
        # Canonicalize input data into JSON with sorted keys
        hash_input = json.dumps(
            {
                "type": self.type.value,
                "organization_id": self.organization_id,
                "config": self.config,
            },
            sort_keys=True,  # Ensure stable ordering
            separators=(",", ":"),  # Minimize serialization differences
        )

        # Compute SHA-256 hash
        hash_digest = hashlib.sha256(hash_input.encode("utf-8")).digest()

        # Convert the digest to an integer for compatibility with Python's hash requirements
        return str(int.from_bytes(hash_digest, byteorder="big"))


class SandboxConfigCreate(LettaBase):
    config: Union[LocalSandboxConfig, E2BSandboxConfig] = Field(..., description="The configuration for the sandbox.")


class SandboxConfigUpdate(LettaBase):
    """Pydantic model for updating SandboxConfig fields."""

    config: Union[LocalSandboxConfig, E2BSandboxConfig] = Field(None, description="The JSON configuration data for the sandbox.")


# Environment Variable
class SandboxEnvironmentVariableBase(OrmMetadataBase):
    __id_prefix__ = "sandbox-env"


class SandboxEnvironmentVariable(SandboxEnvironmentVariableBase):
    id: str = SandboxEnvironmentVariableBase.generate_id_field()
    key: str = Field(..., description="The name of the environment variable.")
    value: str = Field(..., description="The value of the environment variable.")
    description: Optional[str] = Field(None, description="An optional description of the environment variable.")
    sandbox_config_id: str = Field(..., description="The ID of the sandbox config this environment variable belongs to.")
    organization_id: Optional[str] = Field(None, description="The ID of the organization this environment variable belongs to.")


class SandboxEnvironmentVariableCreate(LettaBase):
    key: str = Field(..., description="The name of the environment variable.")
    value: str = Field(..., description="The value of the environment variable.")
    description: Optional[str] = Field(None, description="An optional description of the environment variable.")


class SandboxEnvironmentVariableUpdate(LettaBase):
    """Pydantic model for updating SandboxEnvironmentVariable fields."""

    key: Optional[str] = Field(None, description="The name of the environment variable.")
    value: Optional[str] = Field(None, description="The value of the environment variable.")
    description: Optional[str] = Field(None, description="An optional description of the environment variable.")
