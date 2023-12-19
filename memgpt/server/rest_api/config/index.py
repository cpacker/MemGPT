from fastapi import APIRouter, Depends, HTTPException

from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ConfigRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user issuing the command.")


class ConfigResponse(BaseModel):
    config: dict = Field(..., description="The server configuration object.")


def setup_config_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/config", tags=["config"], response_model=ConfigResponse)
    def get_server_config(user_id: ConfigRequest = Depends()):
        """
        Retrieve the base configuration for the server.
        """
        interface.clear()
        response = server.get_server_config(user_id=user_id)
        return ConfigResponse(config=response)

    return router
