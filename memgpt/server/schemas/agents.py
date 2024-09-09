from datetime import datetime
from typing import Annotated, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from memgpt.orm.enums import MessageRoleType
from memgpt.schemas.agent import AgentState
from memgpt.schemas.usage import MemGPTUsageStatistics


class AgentCommandRequest(BaseModel):
    command: str = Field(..., description="The command to be executed by the agent.")


class AgentCommandResponse(BaseModel):
    response: str = Field(..., description="The result of the executed command.")


class AgentRenameRequest(BaseModel):
    agent_name: Annotated[str, StringConstraints(min_length=1, max_length=50, pattern="^[A-Za-z0-9 _-]+$")] = Field(
        ..., description="New name for the agent."
    )


class GetAgentResponse(BaseModel):
    # config: dict = Field(..., description="The agent configuration object.")
    agent_state: AgentState = Field(..., description="The state of the agent.")
    sources: List[str] = Field(..., description="The list of data sources associated with the agent.")
    last_run_at: Optional[int] = Field(None, description="The unix timestamp of when the agent was last run.")


class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    # TODO make return type List[AgentState]
    agents: List[dict] = Field(..., description="List of agent configurations.")


class CreateAgentRequest(BaseModel):
    # TODO: modify this (along with front end)
    config: dict = Field(..., description="The agent configuration object.")


class CreateAgentResponse(BaseModel):
    agent_state: "AgentState" = Field(..., description="The state of the newly created agent.")


class CoreMemory(BaseModel):
    human: str | None = Field(None, description="Human element of the core memory.")
    persona: str | None = Field(None, description="Persona element of the core memory.")


class GetAgentMemoryResponse(BaseModel):
    core_memory: CoreMemory = Field(..., description="The state of the agent's core memory.")
    recall_memory: int = Field(..., description="Size of the agent's recall memory.")
    archival_memory: int = Field(..., description="Size of the agent's archival memory.")


# NOTE not subclassing CoreMemory since in the request both field are optional
class UpdateAgentMemoryRequest(BaseModel):
    human: str = Field(None, description="Human element of the core memory.")
    persona: str = Field(None, description="Persona element of the core memory.")


class UpdateAgentMemoryResponse(BaseModel):
    old_core_memory: CoreMemory = Field(..., description="The previous state of the agent's core memory.")
    new_core_memory: CoreMemory = Field(..., description="The updated state of the agent's core memory.")


class ArchivalMemoryObject(BaseModel):
    # TODO move to models/pydantic_models, or inherent from data_types Record
    id: "UUID" = Field(..., description="Unique identifier for the memory object inside the archival memory store.")
    contents: str = Field(..., description="The memory contents.")


class GetAgentArchivalMemoryResponse(BaseModel):
    # TODO: make this List[Passage] instead
    archival_memory: List[ArchivalMemoryObject] = Field(..., description="A list of all memory objects in archival memory.")


class InsertAgentArchivalMemoryRequest(BaseModel):
    content: str = Field(..., description="The memory contents to insert into archival memory.")


class InsertAgentArchivalMemoryResponse(BaseModel):
    ids: List[str] = Field(
        ..., description="Unique identifier for the new archival memory object. May return multiple ids if insert contents are chunked."
    )


class DeleteAgentArchivalMemoryRequest(BaseModel):
    id: str = Field(..., description="Unique identifier for the new archival memory object.")


class UserMessageRequest(BaseModel):
    message: str = Field(..., description="The message content to be processed by the agent.")
    name: Optional[str] = Field(default=None, description="Name of the message request sender")
    role: MessageRoleType = Field(default=MessageRoleType.user, description="Role of the message sender (either 'user' or 'system')")
    stream_steps: bool = Field(
        default=False, description="Flag to determine if the response should be streamed. Set to True for streaming agent steps."
    )
    stream_tokens: bool = Field(
        default=False,
        description="Flag to determine if individual tokens should be streamed. Set to True for token streaming (requires stream_steps = True).",
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp to tag the message with (in ISO format). If null, timestamp will be created server-side on receipt of message.",
    )
    stream: bool = Field(
        default=False,
        description="Legacy flag for old streaming API, will be deprecrated in the future.",
        deprecated=True,
    )

    # @validator("timestamp", pre=True, always=True)
    # def validate_timestamp(cls, value: Optional[datetime]) -> Optional[datetime]:
    #    if value is None:
    #        return value  # If the timestamp is None, just return None, implying default handling to set server-side

    #    if not isinstance(value, datetime):
    #        raise TypeError("Timestamp must be a datetime object with timezone information.")

    #    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
    #        raise ValueError("Timestamp must be timezone-aware.")

    #    # Convert timestamp to UTC if it's not already in UTC
    #    if value.tzinfo.utcoffset(value) != timezone.utc.utcoffset(value):
    #        value = value.astimezone(timezone.utc)

    #    return value


class UserMessageResponse(BaseModel):
    messages: List[dict] = Field(..., description="List of messages generated by the agent in response to the received message.")
    usage: MemGPTUsageStatistics = Field(..., description="Usage statistics for the completion.")


class GetAgentMessagesRequest(BaseModel):
    start: int = Field(..., description="Message index to start on (reverse chronological).")
    count: int = Field(..., description="How many messages to retrieve.")


class GetAgentMessagesCursorRequest(BaseModel):
    before: Optional["UUID"] = Field(..., description="Message before which to retrieve the returned messages.")
    limit: int = Field(..., description="Maximum number of messages to retrieve.")


class GetAgentMessagesResponse(BaseModel):
    messages: list = Field(..., description="List of message objects.")
