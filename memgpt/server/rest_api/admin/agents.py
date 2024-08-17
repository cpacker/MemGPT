from fastapi import APIRouter

from memgpt.server.rest_api.agents.index import ListAgentsResponse
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_admin_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents", tags=["agents"], response_model=ListAgentsResponse)
    def get_all_agents():
        """
        Get a list of all agents in the database
        """
        interface.clear()
        agents_data = server.list_agents_legacy()

        return ListAgentsResponse(**agents_data)

    return router
