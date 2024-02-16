import uuid
import re
from functools import partial

from fastapi import APIRouter, Body, Depends, Query, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.auth_token import get_current_user

router = APIRouter()


class AgentConfigRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier of the agent whose config is requested.")


class AgentRenameRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier of the agent whose config is requested.")
    agent_name: str = Field(..., description="New name for the agent.")


class AgentConfigResponse(BaseModel):
    config: dict = Field(..., description="The agent configuration object.")


def validate_agent_name(name: str) -> str:
    """Validate the requested new agent name (prevent bad inputs)"""

    # Length check
    if not (1 <= len(name) <= 50):
        raise HTTPException(status_code=400, detail="Name length must be between 1 and 50 characters.")

    # Regex for allowed characters (alphanumeric, spaces, hyphens, underscores)
    if not re.match("^[A-Za-z0-9 _-]+$", name):
        raise HTTPException(status_code=400, detail="Name contains invalid characters.")

    # Further checks can be added here...
    # TODO

    return name


def setup_agents_config_router(server: SyncServer, interface: QueuingInterface):
    get_current_user_with_server = partial(get_current_user, server)

    @router.get("/agents/config", tags=["agents"], response_model=AgentConfigResponse)
    def get_agent_config(
        agent_id: str = Query(..., description="Unique identifier of the agent whose config is requested."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the configuration for a specific agent.

        This endpoint fetches the configuration details for a given agent, identified by the user and agent IDs.
        """
        request = AgentConfigRequest(agent_id=agent_id)

        agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        interface.clear()
        config = server.get_agent_config(user_id=user_id, agent_id=agent_id)
        return AgentConfigResponse(config=config)

    @router.patch("/agents/rename", tags=["agents"], response_model=AgentConfigResponse)
    def update_agent_name(
        request: AgentRenameRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Updates the name of a specific agent.

        This changes the name of the agent in the database but does NOT edit the agent's persona.
        """
        agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        valid_name = validate_agent_name(request.agent_name)

        interface.clear()
        try:
            config = server.rename_agent(user_id=user_id, agent_id=agent_id, new_agent_name=valid_name)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return AgentConfigResponse(config=config)

    @router.delete("/agents", tags=["agents"])
    def delete_agent(
        agent_id: str = Query(..., description="Unique identifier of the agent to be deleted."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Delete an agent.
        """
        request = AgentConfigRequest(agent_id=agent_id)

        agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        interface.clear()
        try:
            server.delete_agent(user_id=user_id, agent_id=agent_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Agent agent_id={agent_id} successfully deleted"})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
