from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class CreateAgentConfig(BaseModel):
    user_id: str
    config: dict


def setup_config_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/config")
    def get_server_config(user_id: str):
        interface.clear()
        return server.get_server_config(user_id=user_id)

    return router
