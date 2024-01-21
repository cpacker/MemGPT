import uuid
from typing import List

from fastapi import APIRouter, Depends, Body, Query, HTTPException
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
    agent_id: uuid.UUID = Field(..., description="Unique identifier of the newly created agent.")


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents", tags=["agents"], response_model=ListAgentsResponse)
    def list_agents(user_id: str = Query(..., description="Unique identifier of the user.")):
        """
        List all agents associated with a given user.

        This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
        """
        request = ListAgentsRequest(user_id=user_id)

        # TODO remove once chatui adds user selection / pulls user from config
        request.user_id = None if request.user_id == "null" else request.user_id

        user_id = uuid.UUID(request.user_id) if request.user_id else None

        interface.clear()
        agents_data = server.list_agents(user_id=user_id)
        return ListAgentsResponse(**agents_data)

    @router.post("/agents", tags=["agents"], response_model=CreateAgentResponse)
    def create_agent(request: CreateAgentRequest = Body(...)):
        """
        Create a new agent with the specified configuration.
        """
        interface.clear()

        # TODO remove once chatui adds user selection / pulls user from config
        request.user_id = None if request.user_id == "null" else request.user_id

        try:
            user_id = uuid.UUID(request.user_id) if request.user_id else None
            agent_state = server.create_agent(user_id=user_id, agent_config=request.config)
            return CreateAgentResponse(agent_id=agent_state.id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
