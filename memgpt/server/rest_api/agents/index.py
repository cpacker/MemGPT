import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.agent import AgentState, CreateAgent
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents", tags=["agents"], response_model=List[AgentState])
    def list_agents(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        List all agents associated with a given user.

        This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
        """
        interface.clear()
        agents_data = server.list_agents(user_id=user_id)
        return agents_data

    @router.post("/agents", tags=["agents"], response_model=AgentState)
    def create_agent(
        request: CreateAgent = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Create a new agent with the specified configuration.
        """
        interface.clear()

        try:
            agent_state = server.create_agent(request, user_id=user_id)
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail=str(e))

        return agent_state

    return router
