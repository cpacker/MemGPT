import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.letta_base import LettaBase
from letta.schemas.memory import Memory
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics


class BaseAgent(LettaBase, validate_assignment=True):
    __id_prefix__ = "agent"
    description: Optional[str] = Field(None, description="The description of the agent.")

    # metadata
    metadata_: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")
    user_id: Optional[str] = Field(None, description="The user id of the agent.")


class AgentState(BaseAgent):
    """
    Representation of an agent's state. This is the state of the agent at a given time, and is persisted in the DB backend. The state has all the information needed to recreate a persisted agent.

    Parameters:
        id (str): The unique identifier of the agent.
        name (str): The name of the agent (must be unique to the user).
        created_at (datetime): The datetime the agent was created.
        message_ids (List[str]): The ids of the messages in the agent's in-context memory.
        memory (Memory): The in-context memory of the agent.
        tools (List[str]): The tools used by the agent. This includes any memory editing functions specified in `memory`.
        system (str): The system prompt used by the agent.
        llm_config (LLMConfig): The LLM configuration used by the agent.
        embedding_config (EmbeddingConfig): The embedding configuration used by the agent.

    """

    id: str = BaseAgent.generate_id_field()
    name: str = Field(..., description="The name of the agent.")
    created_at: datetime = Field(..., description="The datetime the agent was created.", default_factory=datetime.now)

    # in-context memory
    message_ids: Optional[List[str]] = Field(default=None, description="The ids of the messages in the agent's in-context memory.")
    memory: Memory = Field(default_factory=Memory, description="The in-context memory of the agent.")

    # tools
    tools: List[str] = Field(..., description="The tools used by the agent.")

    # system prompt
    system: str = Field(..., description="The system prompt used by the agent.")

    # llm information
    llm_config: LLMConfig = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the agent.")


class CreateAgent(BaseAgent):
    # all optional as server can generate defaults
    name: Optional[str] = Field(None, description="The name of the agent.")
    message_ids: Optional[List[uuid.UUID]] = Field(None, description="The ids of the messages in the agent's in-context memory.")
    memory: Optional[Memory] = Field(None, description="The in-context memory of the agent.")
    tools: Optional[List[str]] = Field(None, description="The tools used by the agent.")
    system: Optional[str] = Field(None, description="The system prompt used by the agent.")
    llm_config: Optional[LLMConfig] = Field(None, description="The LLM configuration used by the agent.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the agent.")

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """Validate the requested new agent name (prevent bad inputs)"""

        import re

        if not name:
            # don't check if not provided
            return name

        # TODO: this check should also be added to other model (e.g. User.name)
        # Length check
        if not (1 <= len(name) <= 50):
            raise ValueError("Name length must be between 1 and 50 characters.")

        # Regex for allowed characters (alphanumeric, spaces, hyphens, underscores)
        if not re.match("^[A-Za-z0-9 _-]+$", name):
            raise ValueError("Name contains invalid characters.")

        # Further checks can be added here...
        # TODO

        return name


class UpdateAgentState(BaseAgent):
    id: str = Field(..., description="The id of the agent.")
    name: Optional[str] = Field(None, description="The name of the agent.")
    tools: Optional[List[str]] = Field(None, description="The tools used by the agent.")
    system: Optional[str] = Field(None, description="The system prompt used by the agent.")
    llm_config: Optional[LLMConfig] = Field(None, description="The LLM configuration used by the agent.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the agent.")

    # TODO: determine if these should be editable via this schema?
    message_ids: Optional[List[str]] = Field(None, description="The ids of the messages in the agent's in-context memory.")
    memory: Optional[Memory] = Field(None, description="The in-context memory of the agent.")


class AgentStepResponse(BaseModel):
    # TODO remove support for list of dicts
    messages: Union[List[Message], List[dict]] = Field(..., description="The messages generated during the agent's step.")
    heartbeat_request: bool = Field(..., description="Whether the agent requested a heartbeat (i.e. follow-up execution).")
    function_failed: bool = Field(..., description="Whether the agent step ended because a function call failed.")
    in_context_memory_warning: bool = Field(
        ..., description="Whether the agent step ended because the in-context memory is near its limit."
    )
    usage: UsageStatistics = Field(..., description="Usage statistics of the LLM call during the agent's step.")
