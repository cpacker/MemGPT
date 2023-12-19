from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class AgentConfigRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user issuing the command.")
    agent_id: str = Field(..., description="Identifier of the agent on which the command will be executed.")


class AgentConfigResponse(BaseModel):
    config: dict = Field(..., description="The agent configuration object.")


def setup_agents_config_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents/config")
    def get_agent_config(request: AgentConfigRequest = Depends()):
        """
        Retrieve the configuration for a specific agent.

        This endpoint fetches the configuration details for a given agent, identified by the user and agent IDs.
        It clears any existing interface states before retrieving the configuration.

        :param request: AgentConfigRequest object containing the user_id and agent_id.
        :return: An AgentConfigResponse object containing the configuration details of the requested agent.
        """
        interface.clear()
        config = server.get_agent_config(user_id=request.user_id, agent_id=request.agent_id)
        return AgentConfigResponse(config=config)

    return router
