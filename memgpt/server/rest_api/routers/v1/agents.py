from fastapi import APIRouter, Depends, HTTPException

from memgpt.server.rest_api.utils import get_current_user, get_current_interface, get_memgpt_server
from memgpt.server.schemas.agents import AgentCommandResponse

if TYPE_CHECKING:
    from uuid import UUID
    from memgpt.orm.user import User
    from memgpt.server.schemas.agents import AgentCommandRequest
    from memgpt.server.rest_api.interface import QueuingInterface
    from memgpt.server.server import SyncServer

router = APIRouter(prefix="/agents", tags=["agents"])


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
