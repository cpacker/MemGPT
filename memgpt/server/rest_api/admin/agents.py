from typing import List

from fastapi import APIRouter

from memgpt.schemas.agent import AgentState
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_admin_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents", tags=["agents"], response_model=List[AgentState])
    def get_all_agents():
        """
        Get a list of all agents in the database
        """
        interface.clear()
        return server.list_agents()

    return router
