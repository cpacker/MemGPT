from functools import partial
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from memgpt.schemas.memory import (
    ArchivalMemorySummary,
    CreateArchivalMemory,
    Memory,
    RecallMemorySummary,
)
from memgpt.schemas.message import Message
from memgpt.schemas.passage import Passage
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_memory_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/memory/messages", tags=["agents"], response_model=List[Message])
    def get_agent_in_context_messages(
        agent_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the messages in the context of a specific agent.
        """
        interface.clear()
        return server.get_in_context_messages(agent_id=agent_id)

    @router.get("/agents/{agent_id}/memory", tags=["agents"], response_model=Memory)
    def get_agent_memory(
        agent_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the memory state of a specific agent.

        This endpoint fetches the current memory state of the agent identified by the user ID and agent ID.
        """
        interface.clear()
        return server.get_agent_memory(agent_id=agent_id)

    @router.post("/agents/{agent_id}/memory", tags=["agents"], response_model=Memory)
    def update_agent_memory(
        agent_id: str,
        request: Dict = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Update the core memory of a specific agent.

        This endpoint accepts new memory contents (human and persona) and updates the core memory of the agent identified by the user ID and agent ID.
        """
        interface.clear()
        memory = server.update_agent_core_memory(user_id=user_id, agent_id=agent_id, new_memory_contents=request)
        return memory

    @router.get("/agents/{agent_id}/memory/recall", tags=["agents"], response_model=RecallMemorySummary)
    def get_agent_recall_memory_summary(
        agent_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the summary of the recall memory of a specific agent.
        """
        interface.clear()
        return server.get_recall_memory_summary(agent_id=agent_id)

    @router.get("/agents/{agent_id}/memory/archival", tags=["agents"], response_model=ArchivalMemorySummary)
    def get_agent_archival_memory_summary(
        agent_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the summary of the archival memory of a specific agent.
        """
        interface.clear()
        return server.get_archival_memory_summary(agent_id=agent_id)

    # @router.get("/agents/{agent_id}/archival/all", tags=["agents"], response_model=List[Passage])
    # def get_agent_archival_memory_all(
    #    agent_id: str,
    #    user_id: str = Depends(get_current_user_with_server),
    # ):
    #    """
    #    Retrieve the memories in an agent's archival memory store (non-paginated, returns all entries at once).
    #    """
    #    interface.clear()
    #    return server.get_all_archival_memories(user_id=user_id, agent_id=agent_id)

    @router.get("/agents/{agent_id}/archival", tags=["agents"], response_model=List[Passage])
    def get_agent_archival_memory(
        agent_id: str,
        after: Optional[int] = Query(None, description="Unique ID of the memory to start the query range at."),
        before: Optional[int] = Query(None, description="Unique ID of the memory to end the query range at."),
        limit: Optional[int] = Query(None, description="How many results to include in the response."),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the memories in an agent's archival memory store (paginated query).
        """
        interface.clear()
        return server.get_agent_archival_cursor(
            user_id=user_id,
            agent_id=agent_id,
            after=after,
            before=before,
            limit=limit,
        )

    @router.post("/agents/{agent_id}/archival", tags=["agents"], response_model=List[Passage])
    def insert_agent_archival_memory(
        agent_id: str,
        request: CreateArchivalMemory = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Insert a memory into an agent's archival memory store.
        """
        interface.clear()
        return server.insert_archival_memory(user_id=user_id, agent_id=agent_id, memory_contents=request.text)

    @router.delete("/agents/{agent_id}/archival/{memory_id}", tags=["agents"])
    def delete_agent_archival_memory(
        agent_id: str,
        memory_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Delete a memory from an agent's archival memory store.
        """
        # TODO: should probably return a `Passage`
        interface.clear()
        try:
            server.delete_archival_memory(user_id=user_id, agent_id=agent_id, memory_id=memory_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Memory id={memory_id} successfully deleted"})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
