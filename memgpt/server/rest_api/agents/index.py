from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.agent import AgentState, CreateAgent, UpdateAgentState
from memgpt.schemas.source import Source
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents", tags=["agents"], response_model=List[AgentState])
    def list_agents(
        user_id: str = Depends(get_current_user_with_server),
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
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Create a new agent with the specified configuration.
        """
        interface.clear()

        agent_state = server.create_agent(request, user_id=user_id)
        return agent_state

    @router.post("/agents/{agent_id}", tags=["agents"], response_model=AgentState)
    def update_agent(
        agent_id: str,
        request: UpdateAgentState = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """Update an exsiting agent"""
        interface.clear()
        try:
            # TODO: should id be moved out of UpdateAgentState?
            agent_state = server.update_agent(request, user_id=user_id)
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail=str(e))

        return agent_state

    @router.get("/agents/{agent_id}", tags=["agents"], response_model=AgentState)
    def get_agent_state(
        agent_id: str = None,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get the state of the agent.
        """

        interface.clear()
        if not server.ms.get_agent(user_id=user_id, agent_id=agent_id):
            # agent does not exist
            raise HTTPException(status_code=404, detail=f"Agent agent_id={agent_id} not found.")

        return server.get_agent_state(user_id=user_id, agent_id=agent_id)

    @router.delete("/agents/{agent_id}", tags=["agents"])
    def delete_agent(
        agent_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Delete an agent.
        """
        # agent_id = str(agent_id)

        interface.clear()
        try:
            server.delete_agent(user_id=user_id, agent_id=agent_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.get("/agents/{agent_id}/sources", tags=["agents"], response_model=List[Source])
    def get_agent_sources(
        agent_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get the sources associated with an agent.
        """
        interface.clear()
        return server.list_attached_sources(agent_id)

    return router
