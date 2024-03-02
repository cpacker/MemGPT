import uuid
from functools import partial

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ConfigResponse(BaseModel):
    config: dict = Field(..., description="The server configuration object.")
    defaults: dict = Field(..., description="The defaults for the configuration.")


def setup_config_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/config", tags=["config"], response_model=ConfigResponse)
    def get_server_config(user_id: uuid.UUID = Depends(get_current_user_with_server)):
        """
        Retrieve the base configuration for the server.
        """
        interface.clear()
        response = server.get_server_config(include_defaults=True)
        return ConfigResponse(config=response["config"], defaults=response["defaults"])

    return router
