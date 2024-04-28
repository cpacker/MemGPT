import re
import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import (
    AgentStateModel,
    EmbeddingConfigModel,
    LLMConfigModel,
)
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class AgentRenameRequest(BaseModel):
    agent_name: str = Field(..., description="New name for the agent.")


class GetAgentResponse(BaseModel):
    # config: dict = Field(..., description="The agent configuration object.")
    agent_state: AgentStateModel = Field(..., description="The state of the agent.")
    sources: List[str] = Field(..., description="The list of data sources associated with the agent.")
    last_run_at: Optional[int] = Field(None, description="The unix timestamp of when the agent was last run.")


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


def setup_agents_config_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/config", tags=["agents"], response_model=GetAgentResponse)
    def get_agent_config(
        agent_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the configuration for a specific agent.

        This endpoint fetches the configuration details for a given agent, identified by the user and agent IDs.
        """

        interface.clear()
        if not server.ms.get_agent(user_id=user_id, agent_id=agent_id):
            # agent does not exist
            raise HTTPException(status_code=404, detail=f"Agent agent_id={agent_id} not found.")

        agent_state = server.get_agent_config(user_id=user_id, agent_id=agent_id)
        # get sources
        attached_sources = server.list_attached_sources(agent_id=agent_id)

        # configs
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        return GetAgentResponse(
            agent_state=AgentStateModel(
                id=agent_state.id,
                name=agent_state.name,
                user_id=agent_state.user_id,
                preset=agent_state.preset,
                persona=agent_state.persona,
                human=agent_state.human,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=int(agent_state.created_at.timestamp()),
                functions_schema=agent_state.state["functions"],  # TODO: this is very error prone, jsut lookup the preset instead
            ),
            last_run_at=None,  # TODO
            sources=attached_sources,
        )

    @router.patch("/agents/{agent_id}/rename", tags=["agents"], response_model=GetAgentResponse)
    def update_agent_name(
        agent_id: uuid.UUID,
        request: AgentRenameRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Updates the name of a specific agent.

        This changes the name of the agent in the database but does NOT edit the agent's persona.
        """
        # agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        valid_name = validate_agent_name(request.agent_name)

        interface.clear()
        try:
            agent_state = server.rename_agent(user_id=user_id, agent_id=agent_id, new_agent_name=valid_name)
            # get sources
            attached_sources = server.list_attached_sources(agent_id=agent_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        return GetAgentResponse(
            agent_state=AgentStateModel(
                id=agent_state.id,
                name=agent_state.name,
                user_id=agent_state.user_id,
                preset=agent_state.preset,
                persona=agent_state.persona,
                human=agent_state.human,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=int(agent_state.created_at.timestamp()),
                functions_schema=agent_state.state["functions"],  # TODO: this is very error prone, jsut lookup the preset instead
            ),
            last_run_at=None,  # TODO
            sources=attached_sources,
        )

    @router.delete("/agents/{agent_id}", tags=["agents"])
    def delete_agent(
        agent_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Delete an agent.
        """
        # agent_id = uuid.UUID(agent_id)

        interface.clear()
        try:
            server.delete_agent(user_id=user_id, agent_id=agent_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Agent agent_id={agent_id} successfully deleted"})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
