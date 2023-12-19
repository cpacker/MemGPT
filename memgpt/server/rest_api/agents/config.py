from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class AgentConfigRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user requesting the config.")
    agent_id: str = Field(..., description="Identifier of the agent whose config is requested.")


class AgentConfigResponse(BaseModel):
    config: dict = Field(..., description="The agent configuration object.")


def setup_agents_config_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents/config", tags=["agents"], response_model=AgentConfigResponse)
    def get_agent_config(
        user_id: str = Query(..., description="Unique identifier of the user requesting the config."),
        agent_id: str = Query(..., description="Identifier of the agent whose config is requested."),
    ):
        """
        Retrieve the configuration for a specific agent.

        This endpoint fetches the configuration details for a given agent, identified by the user and agent IDs.
        """
        request = AgentConfigRequest(user_id=user_id, agent_id=agent_id)

        interface.clear()
        config = server.get_agent_config(user_id=request.user_id, agent_id=request.agent_id)
        return AgentConfigResponse(config=config)

    return router
