import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
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
    human: str = Field(None, description="Human element of the core memory.")
    persona: str = Field(None, description="Persona element of the core memory.")


class UpdateAgentMemoryResponse(BaseModel):
    old_core_memory: CoreMemory = Field(..., description="The previous state of the agent's core memory.")
    new_core_memory: CoreMemory = Field(..., description="The updated state of the agent's core memory.")


class ArchivalMemoryObject(BaseModel):
    # TODO move to models/pydantic_models, or inherent from data_types Record
    id: uuid.UUID = Field(..., description="Unique identifier for the memory object inside the archival memory store.")
    contents: str = Field(..., description="The memory contents.")


class GetAgentArchivalMemoryResponse(BaseModel):
    archival_memory: List[ArchivalMemoryObject] = Field(..., description="A list of all memory objects in archival memory.")


class InsertAgentArchivalMemoryRequest(BaseModel):
    content: str = Field(..., description="The memory contents to insert into archival memory.")


class InsertAgentArchivalMemoryResponse(BaseModel):
    ids: List[str] = Field(
        ..., description="Unique identifier for the new archival memory object. May return multiple ids if insert contents are chunked."
    )


class DeleteAgentArchivalMemoryRequest(BaseModel):
    id: str = Field(..., description="Unique identifier for the new archival memory object.")


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
        interface.clear()

        new_memory_contents = {"persona": request.persona, "human": request.human}
        response = server.update_agent_core_memory(user_id=user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)
        return UpdateAgentMemoryResponse(**response)

    @router.get("/agents/{agent_id}/archival/all", tags=["agents"], response_model=GetAgentArchivalMemoryResponse)
    def get_agent_archival_memory_all(
        agent_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the memories in an agent's archival memory store (non-paginated, returns all entries at once).
        """
        interface.clear()
        archival_memories = server.get_all_archival_memories(user_id=user_id, agent_id=agent_id)
        print("archival_memories:", archival_memories)
        archival_memory_objects = [ArchivalMemoryObject(id=passage["id"], contents=passage["contents"]) for passage in archival_memories]
        return GetAgentArchivalMemoryResponse(archival_memory=archival_memory_objects)

    @router.get("/agents/{agent_id}/archival", tags=["agents"], response_model=GetAgentArchivalMemoryResponse)
    def get_agent_archival_memory(
        agent_id: uuid.UUID,
        after: Optional[int] = Query(None, description="Unique ID of the memory to start the query range at."),
        before: Optional[int] = Query(None, description="Unique ID of the memory to end the query range at."),
        limit: Optional[int] = Query(None, description="How many results to include in the response."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the memories in an agent's archival memory store (paginated query).
        """
        interface.clear()
        # TODO need to add support for non-postgres here
        # chroma will throw:
        #     raise ValueError("Cannot run get_all_cursor with chroma")
        _, archival_json_records = server.get_agent_archival_cursor(
            user_id=user_id,
            agent_id=agent_id,
            after=after,
            before=before,
            limit=limit,
        )
        archival_memory_objects = [ArchivalMemoryObject(id=passage["id"], contents=passage["text"]) for passage in archival_json_records]
        return GetAgentArchivalMemoryResponse(archival_memory=archival_memory_objects)

    @router.post("/agents/{agent_id}/archival", tags=["agents"], response_model=InsertAgentArchivalMemoryResponse)
    def insert_agent_archival_memory(
        agent_id: uuid.UUID,
        request: InsertAgentArchivalMemoryRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Insert a memory into an agent's archival memory store.
        """
        interface.clear()
        memory_ids = server.insert_archival_memory(user_id=user_id, agent_id=agent_id, memory_contents=request.content)
        return InsertAgentArchivalMemoryResponse(ids=memory_ids)

    @router.delete("/agents/{agent_id}/archival", tags=["agents"])
    def delete_agent_archival_memory(
        agent_id: uuid.UUID,
        id: str = Query(..., description="Unique ID of the memory to be deleted."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Delete a memory from an agent's archival memory store.
        """
        interface.clear()
        try:
            memory_id = uuid.UUID(id)
            server.delete_archival_memory(user_id=user_id, agent_id=agent_id, memory_id=memory_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Memory id={memory_id} successfully deleted"})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
