from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, JSONResponse, status

from memgpt.server.rest_api.utils import get_current_user, get_current_interface, get_memgpt_server
from memgpt.models.pydantic_models import  AgentStateModel, EmbeddingConfigModel, LLMConfigModel
from memgpt.server.schemas.agents import AgentCommandResponse, GetAgentResponse, AgentRenameRequest, CreateAgentRequest, CreateAgentResponse, ListAgentsResponse

if TYPE_CHECKING:
    from uuid import UUID
    from memgpt.orm.user import User
    from memgpt.server.schemas.agents import AgentCommandRequest
    from memgpt.server.rest_api.interface import QueuingInterface
    from memgpt.server.server import SyncServer

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/", response_model=ListAgentsResponse)
def list_agents(
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all agents associated with a given user.

    This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
    """
    interface.clear()
    return server.list_agents(user_id=actor._id)

@router.post(response_model=CreateAgentResponse)
def create_agent(
    agent: CreateAgentRequest,
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
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
    request.config["preset"] if ("preset" in request.config and request.config["preset"]) else settings.preset
    tool_names = request.config["function_names"]
    metadata = request.config["metadata"] if "metadata" in request.config else {}
    metadata["human"] = human_name
    metadata["persona"] = persona_name

    # TODO: remove this -- should be added based on create agent fields
    if isinstance(tool_names, str):  # TODO: fix this on clinet side?
        tool_names = tool_names.split(",")
    if tool_names is None or tool_names == "":
        tool_names = []
    for name in BASE_TOOLS:  # TODO: remove this
        if name not in tool_names:
            tool_names.append(name)
    assert isinstance(tool_names, list), "Tool names must be a list of strings."

    # TODO: eventually remove this - should support general memory at the REST endpoint
    memory = ChatMemory(persona=persona, human=human)

    try:
        agent_state = server.create_agent(
            user_id=user_id,
            # **request.config
            # TODO turn into a pydantic model
            name=request.config["name"],
            memory=memory,
            # persona_name=persona_name,
            # human_name=human_name,
            # persona=persona,
            # human=human,
            # llm_config=LLMConfigModel(
            # model=request.config['model'],
            # )
            # tools
            tools=tool_names,
            metadata=metadata,
            # function_names=request.config["function_names"].split(",") if "function_names" in request.config else None,
        )
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        return CreateAgentResponse(
            agent_state=AgentStateModel(
                id=agent_state.id,
                name=agent_state.name,
                user_id=agent_state.user_id,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=int(agent_state.created_at.timestamp()),
                tools=tool_names,
                system=agent_state.system,
                metadata=agent_state._metadata,
            ),
            preset=PresetModel(  # TODO: remove (placeholder to avoid breaking frontend)
                name="dummy_preset",
                id=agent_state.id,
                user_id=agent_state.user_id,
                description="",
                created_at=agent_state.created_at,
                system=agent_state.system,
                persona="",
                human="",
                functions_schema=[],
            ),
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/command")
def run_command(
    agent_id: "UUID",
    command: "AgentCommandRequest",
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Execute a command on a specified agent.

    This endpoint receives a command to be executed on an agent. It uses the user and agent identifiers to authenticate and route the command appropriately.

    Raises an HTTPException for any processing errors.
    """
    interface.clear()
    response = server.run_command(user_id=actor._id,
                                  agent_id=agent_id,
                                  command=command.command)

    return AgentCommandResponse(response=response)


@router.get("/{agent_id}/config")
def get_agent_config(
    agent_id: "UUID",
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Retrieve the configuration for a specific agent.

    This endpoint fetches the configuration details for a given agent, identified by the user and agent IDs.
    """

    interface.clear()
    if not server.ms.get_agent(user_id=actor._id, agent_id=agent_id):
        # agent does not exist
        raise HTTPException(status_code=404, detail=f"Agent agent_id={agent_id} not found.")

    agent_state = server.get_agent_config(user_id=actor._id, agent_id=agent_id)
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
            llm_config=llm_config,
            embedding_config=embedding_config,
            state=agent_state.state,
            created_at=int(agent_state.created_at.timestamp()),
            tools=agent_state.tools,
            system=agent_state.system,
            metadata=agent_state._metadata,
        ),
        last_run_at=None,  # TODO
        sources=attached_sources,
    )

@router.patch("/agents/{agent_id}/rename", tags=["agents"], response_model=GetAgentResponse)
def update_agent_name(
    agent_id: uuid.UUID,
    agent_rename: AgentRenameRequest,
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Updates the name of a specific agent.

    This changes the name of the agent in the database but does NOT edit the agent's persona.
    """
    valid_name = agent_rename.agent_name

    interface.clear()
    agent_state = server.rename_agent(user_id=actor._id, agent_id=agent_id, new_agent_name=valid_name)
    # get sources
    attached_sources = server.list_attached_sources(agent_id=agent_id)
    llm_config = LLMConfigModel(**vars(agent_state.llm_config))
    embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

    return GetAgentResponse(
        agent_state=AgentStateModel(
            id=agent_state.id,
            name=agent_state.name,
            user_id=agent_state.user_id,
            llm_config=llm_config,
            embedding_config=embedding_config,
            state=agent_state.state,
            created_at=int(agent_state.created_at.timestamp()),
            tools=agent_state.tools,
            system=agent_state.system,
        ),
        last_run_at=None,  # TODO
        sources=attached_sources,
    )

@router.delete("/agents/{agent_id}", tags=["agents"])
def delete_agent(
    agent_id: "UUID",
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Delete an agent.
    """
    # agent_id = uuid.UUID(agent_id)

    interface.clear()
    server.delete_agent(user_id=actor._id, agent_id=agent_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Agent agent_id={agent_id} successfully deleted"})