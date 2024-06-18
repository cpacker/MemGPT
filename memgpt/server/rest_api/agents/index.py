import datetime
import uuid
from functools import partial
from http.client import HTTPException
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi import status as stat
from pydantic import BaseModel, Field

# krishna1
from rich import print

from memgpt.data_types import AgentState, LLMConfig
from memgpt.models.pydantic_models import (
    AgentStateModel,
    AgentStateWithSourcesModel,
    EmbeddingConfigModel,
    LLMConfigModel,
    PresetModel,
)
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.streaming_interface import (
    StreamingRefreshCLIInterface as stream_interface,  # for printing to terminal
)

router = APIRouter()


class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    # TODO return - presets: List[PresetModel]
    agents: List[AgentStateWithSourcesModel] = Field(..., description="List of agent configurations.")


class CreateAgentRequest(BaseModel):
    config: dict = Field(..., description="The agent configuration object.")


class UpdateAgentRequest(BaseModel):
    agent_id: int = Field(..., description="ID of agent.")
    agent_name: str = Field(..., description="Name of agent.")
    persona: str = Field(..., description="Persona of agent.")
    human: str = Field(..., description="Info about the person that the agent interacts with.")
    model: Optional[str] = Field(..., description="Model name.")
    context_window: Optional[int] = Field(..., description="Length of context window.")
    model_wrapper: Optional[str] = Field(..., description="Wrapper around model.")
    model_endpoint: Optional[str] = Field(..., description="Endpoint to reach model.")
    model_endpoint_type: Optional[str] = Field(..., description="Model endpoint type.")


class AgentIdRequest(BaseModel):
    agent_id: int = Field(..., description="ID of agent.")


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
        # krishna
        print("index agents_data: ", agents_data["agents"][0])
        return ListAgentsResponse(**agents_data)

    @router.post("/agents/create", tags=["agents"], response_model=CreateAgentResponse)
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

    @router.post("/agents/update", tags=["agents"])
    def update_agent(
        request: UpdateAgentRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Updates the agent given the modified AgentState object
        """
        # first, get agent (source: `get_agent_config` in rest_api/agents/config.py)
        interface.clear()
        agent_id = uuid.UUID(int=request.agent_id)
        if not server.ms.get_agent(user_id=user_id, agent_id=agent_id, agent_name=request.agent_name):
            # agent does not exist
            raise HTTPException(status_code=stat.HTTP_404_NOT_FOUND, detail=f"Agent agent_id={agent_id} not found.")

        agent_state = server.get_agent_config(user_id=user_id, agent_id=agent_id, agent_name=request.agent_name)

        # configs
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        updated_agent = (
            AgentState(
                id=agent_state.id,
                name=agent_state.name,
                user_id=agent_state.user_id,
                preset=agent_state.preset,
                persona=request.persona,
                human=request.human,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=agent_state.created_at,
            ),
        )
        # krishna1
        # print("update_agent value: ", updated_agent)
        updated_agent = updated_agent[0]
        updated_agent.llm_config.model = request.model
        updated_agent.llm_config.context_window = request.context_window
        updated_agent.llm_config.model_wrapper = request.model_wrapper
        updated_agent.llm_config.model_endpoint = request.model_endpoint
        updated_agent.llm_config.model_endpoint_type = request.model_endpoint_type

        # then, perform update
        interface.clear()
        server.update_agent(agent_state=updated_agent)

    # krishna
    # @router.post("/agents/save", tags=["agents"])
    # def save_agent(
    #     request: Union[PresetWithMetadata, AgentIdRequest]= Body(...),
    #     user_id: uuid.UUID = Depends(get_current_user_with_server),
    # ):
    #     """
    #     Two Cases:
    #     #1 - save_agent is called during `memgpt run`
    #         the Preset object must be saved and we do not have access to agent_id, so PresetWithMetadata is used
    #     #2 - save_agent is called in main.py during `run_agent_loop` (and likely other places in the future)
    #         the Agent object was previously instantiated and saved in storage, therefore AgentIdRequest is used
    #     """
    #     # krishna
    #     print("agent index request: ", request)

    #     # obtain agent_state
    #     interface.clear()
    #     if isinstance(request, AgentIdRequest) :
    #         agent_id = uuid.UUID(int=request.agent_id)
    #         agent_name = None
    #     else:
    #         agent_id = None
    #         agent_name = request.agent_name

    #     if not server.ms.get_agent(user_id=user_id, agent_name=agent_name, agent_id=agent_id):
    #         # agent does not exist
    #         raise HTTPException(status_code=404, detail=f"Agent {agent_name} / {request.agent_id} not found.")

    #     agent_state = server.get_agent_config(user_id=user_id, agent_name=agent_name, agent_id=agent_id)

    #     # generate configs
    #     llm_config = LLMConfigModel(**vars(agent_state.llm_config))
    #     embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

    #     # create Agent object from the following (COPY :690 OF CLI.PY):
    #         # AgentState
    #         # configs
    #         # interface()
    #         # metadata (see client.py save_agent)

    #     interface.clear()
    #     memgpt_agent = Agent(
    #             interface=stream_interface(),
    #             name=agent_state.name,
    #             agent_state=agent_state,
    #             created_by=agent_state.user_id,
    #             preset=None if isinstance(request, AgentIdRequest) else request,
    #             llm_config=llm_config,
    #             embedding_config=embedding_config,
    #             first_message_verify_mono=False if isinstance(request, AgentIdRequest) else request.first_message_verify_mono
    #         )
    #     server.save_agent(memgpt_agent)

    @router.post("/agents/save", tags=["agents"])
    def save_agent(
        request: AgentStateModel = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Saves the Agent to metadata store.
        """

        interface.clear()
        agent_state = AgentState(
            name=request.name,
            user_id=request.user_id,
            persona=request.persona,
            human=request.human,
            llm_config=LLMConfig(**request.llm_config.model_dump()),
            embedding_config=EmbeddingConfigModel(**request.embedding_config.model_dump()),
            preset=request.preset,
            id=request.id,
            state=request.state,
            created_at=datetime.datetime.fromtimestamp(request.created_at),
        )
        server.save_agent_using_state(agent_state)

    return router
