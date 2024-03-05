import uuid
from functools import partial

from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel, Field

from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class CoreMemory(BaseModel):
    human: str | None = Field(None, description="Human element of the core memory.")
    persona: str | None = Field(None, description="Persona element of the core memory.")


class GetAgentMemoryResponse(BaseModel):
    core_memory: CoreMemory = Field(..., description="The state of the agent's core memory.")
    recall_memory: int = Field(..., description="Size of the agent's recall memory.")
    archival_memory: int = Field(..., description="Size of the agent's archival memory.")


# NOTE not subclassing CoreMemory since in the request both field are optional
class UpdateAgentMemoryRequest(BaseModel):
    agent_id: str = Field(..., description="The unique identifier of the agent.")
    human: str = Field(None, description="Human element of the core memory.")
    persona: str = Field(None, description="Persona element of the core memory.")


class UpdateAgentMemoryResponse(BaseModel):
    old_core_memory: CoreMemory = Field(..., description="The previous state of the agent's core memory.")
    new_core_memory: CoreMemory = Field(..., description="The updated state of the agent's core memory.")


def setup_agents_memory_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/memory", tags=["agents"], response_model=GetAgentMemoryResponse)
    def get_agent_memory(
        agent_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the memory state of a specific agent.

        This endpoint fetches the current memory state of the agent identified by the user ID and agent ID.
        """
        interface.clear()
        memory = server.get_agent_memory(user_id=user_id, agent_id=agent_id)
        return GetAgentMemoryResponse(**memory)

    @router.post("/agents/{agent_id}/memory", tags=["agents"], response_model=UpdateAgentMemoryResponse)
    def update_agent_memory(
        agent_id: uuid.UUID,
        request: UpdateAgentMemoryRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Update the core memory of a specific agent.

        This endpoint accepts new memory contents (human and persona) and updates the core memory of the agent identified by the user ID and agent ID.
        """
        agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        interface.clear()

        new_memory_contents = {"persona": request.persona, "human": request.human}
        response = server.update_agent_core_memory(user_id=user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)
        return UpdateAgentMemoryResponse(**response)

    return router
