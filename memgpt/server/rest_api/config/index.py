from fastapi import APIRouter, Depends, Query, HTTPException

from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ConfigRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user requesting the config.")


class ConfigResponse(BaseModel):
    config: dict = Field(..., description="The server configuration object.")


def setup_config_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/config", tags=["config"], response_model=ConfigResponse)
    def get_server_config(user_id: str = Query(..., description="Unique identifier of the user requesting the config.")):
        """
        Retrieve the base configuration for the server.
        """
        request = ConfigRequest(user_id=user_id)

        interface.clear()
        response = server.get_server_config(user_id=request.user_id)
        return ConfigResponse(config=response)

    return router
