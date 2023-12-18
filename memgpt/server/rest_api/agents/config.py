from fastapi import APIRouter

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_config_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents/config")
    def get_agent_config(user_id: str, agent_id: str):
        interface.clear()
        return server.get_agent_config(user_id=user_id, agent_id=agent_id)

    return router
