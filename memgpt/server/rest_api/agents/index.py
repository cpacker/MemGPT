import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import (
    AgentStateModel,
    EmbeddingConfigModel,
    LLMConfigModel,
    PresetModel,
)
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    # TODO make return type List[AgentStateModel]
    #      also return - presets: List[PresetModel]
    agents: List[dict] = Field(..., description="List of agent configurations.")


class CreateAgentRequest(BaseModel):
    config: dict = Field(..., description="The agent configuration object.")


class CreateAgentResponse(BaseModel):
    agent_state: AgentStateModel = Field(..., description="The state of the newly created agent.")
    preset: PresetModel = Field(..., description="The preset that the agent was created from.")


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents", tags=["agents"], response_model=ListAgentsResponse)
    def list_agents(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        List all agents associated with a given user.

        This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
        """
        interface.clear()
        agents_data = server.list_agents(user_id=user_id)
        return ListAgentsResponse(**agents_data)

    @router.post("/agents", tags=["agents"], response_model=CreateAgentResponse)
    def create_agent(
        request: CreateAgentRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Create a new agent with the specified configuration.
        """
        interface.clear()

        # try:
        agent_state = server.create_agent(
            user_id=user_id,
            # **request.config
            # TODO turn into a pydantic model
            name=request.config["name"],
            preset=request.config["preset"] if "preset" in request.config else None,
            persona_name=request.config["persona_name"] if "persona_name" in request.config else None,
            human_name=request.config["human_name"] if "human_name" in request.config else None,
            persona=request.config["persona"] if "persona" in request.config else None,
            human=request.config["human"] if "human" in request.config else None,
            # llm_config=LLMConfigModel(
            # model=request.config['model'],
            # )
            function_names=request.config["function_names"].split(",") if "function_names" in request.config else None,
        )
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        # TODO when get_preset returns a PresetModel instead of Preset, we can remove this packing/unpacking line
        preset = server.ms.get_preset(name=agent_state.preset, user_id=user_id)

        return CreateAgentResponse(
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
            preset=PresetModel(
                name=preset.name,
                id=preset.id,
                user_id=preset.user_id,
                description=preset.description,
                created_at=preset.created_at,
                system=preset.system,
                persona=preset.persona,
                human=preset.human,
                functions_schema=preset.functions_schema,
            ),
        )
        # except Exception as e:
        #    print(str(e))
        #    raise HTTPException(status_code=500, detail=str(e))

    return router
