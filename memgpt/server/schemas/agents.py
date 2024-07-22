from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, Field, StringConstraints

from memgpt.models.pydantic_models import AgentStateModel

if TYPE_CHECKING:
    from memgpt.models.pydantic_models import PresetModel


class AgentCommandRequest(BaseModel):
    command: str = Field(..., description="The command to be executed by the agent.")


class AgentCommandResponse(BaseModel):
    response: str = Field(..., description="The result of the executed command.")

class AgentRenameRequest(BaseModel):
    agent_name: str = Field(...,
                            StringConstraints(min_length=1, max_length=50),
                            description="New name for the agent.",
                            pattern="^[A-Za-z0-9 _-]+$")

class GetAgentResponse(BaseModel):
    # config: dict = Field(..., description="The agent configuration object.")
    agent_state: AgentStateModel = Field(..., description="The state of the agent.")
    sources: List[str] = Field(..., description="The list of data sources associated with the agent.")
    last_run_at: Optional[int] = Field(None, description="The unix timestamp of when the agent was last run.")

class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    # TODO make return type List[AgentStateModel]
    #      also return - presets: List[PresetModel]
    agents: List[dict] = Field(..., description="List of agent configurations.")


class CreateAgentRequest(BaseModel):
    # TODO: modify this (along with front end)
    config: dict = Field(..., description="The agent configuration object.")


class CreateAgentResponse(BaseModel):
    agent_state: AgentStateModel = Field(..., description="The state of the newly created agent.")
    preset: PresetModel = Field(..., description="The preset that the agent was created from.")
