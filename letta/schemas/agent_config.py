from typing import Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AgentType(str, Enum):
    base_agent = "base_agent"
    split_thread_agent = "split_thread_agent"


class AgentConfig(BaseModel):
    """
    Configuration for a Letta agent. This object specifies all the information necessary to access an agent to usage with Letta, except for secret keys.

    Attributes:
        agent_type (str): The type of agent.
    """

    agent_type: AgentType = Field(..., description="The type of agent.")

    @classmethod
    def default_config(cls):
        return cls(agent_type=AgentType.base_agent)
