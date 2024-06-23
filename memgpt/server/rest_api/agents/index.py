import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from memgpt.constants import BASE_TOOLS
from memgpt.models.pydantic_models import (
    AgentStateModel,
    EmbeddingConfigModel,
    LLMConfigModel,
    PresetModel,
)
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.settings import settings

router = APIRouter()


class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    # TODO make return type List[AgentStateModel]
    #      also return - presets: List[PresetModel]
    agents: List[dict] = Field(..., description="List of agent configurations.")


class CreateAgentRequest(BaseModel):
    # TODO: modify this (along with front end)
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

        # Parse request
        # TODO: don't just use JSON in the future
        human_name = request.config["human_name"] if "human_name" in request.config else None
        human = request.config["human"] if "human" in request.config else None
        persona_name = request.config["persona_name"] if "persona_name" in request.config else None
        persona = request.config["persona"] if "persona" in request.config else None
        preset = request.config["preset"] if ("preset" in request.config and request.config["preset"]) else settings.default_preset
        tool_names = request.config["function_names"]

        # TODO: remove this -- should be added based on create agent fields
        if isinstance(tool_names, str):  # TODO: fix this on clinet side?
            tool_names = tool_names.split(",")
        if tool_names is None or tool_names == "":
            tool_names = []
        for name in BASE_TOOLS:  # TODO: remove this
            if name not in tool_names:
                tool_names.append(name)
        assert isinstance(tool_names, list), "Tool names must be a list of strings."

        try:
            agent_state = server.create_agent(
                user_id=user_id,
                # **request.config
                # TODO turn into a pydantic model
                name=request.config["name"],
                preset=preset,
                persona_name=persona_name,
                human_name=human_name,
                persona=persona,
                human=human,
                # llm_config=LLMConfigModel(
                # model=request.config['model'],
                # )
                # tools
                tools=tool_names,
                # function_names=request.config["function_names"].split(",") if "function_names" in request.config else None,
            )
            llm_config = LLMConfigModel(**vars(agent_state.llm_config))
            embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

            # TODO when get_preset returns a PresetModel instead of Preset, we can remove this packing/unpacking line
            # TODO: remove
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
                    tools=tool_names,
                    system=agent_state.system,
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
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail=str(e))

    return router
