from typing import List

from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ListAgentsRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user.")


class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    agents: List[dict] = Field(..., description="List of agent configurations.")


class CreateAgentRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user issuing the command.")
    config: dict = Field(..., description="The agent configuration object.")


class CreateAgentResponse(BaseModel):
    agent_id: str = Field(..., description="Unique identifier of the newly created agent.")


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents", tags=["agents"])
    def list_agents(request: ListAgentsRequest = Depends()):
        """
        List all agents associated with a given user.

        This endpoint retrieves a list of all agents and their configurations associated with
        the specified user ID. It clears any existing interface states before listing the agents.

        :param request: ListAgentsRequest object containing the user_id.
        :return: A ListAgentsResponse object containing the number of agents and their configurations.
        """
        interface.clear()
        agents_data = server.list_agents(user_id=request.user_id)
        return ListAgentsResponse(**agents_data)

    @router.post("/agents")
    def create_agent(request: CreateAgentRequest = Body(...)):
        """
        Create a new agent with the specified configuration.

        This endpoint accepts a user ID and agent configuration to create a new agent.
        It clears any existing interface states before creating the agent.

        :param request: CreateAgentRequest object containing the user_id and agent configuration.
        :return: A CreateAgentResponse object containing the ID of the newly created agent.
        """
        interface.clear()
        try:
            agent_id = server.create_agent(user_id=request.user_id, agent_config=request.config)
            return CreateAgentResponse(agent_id=agent_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
