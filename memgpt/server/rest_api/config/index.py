from functools import partial
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.llm_config import LLMConfig
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ConfigResponse(BaseModel):
    config: dict = Field(..., description="The server configuration object.")
    defaults: dict = Field(..., description="The defaults for the configuration.")


def setup_config_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/config/llm", tags=["config"], response_model=List[LLMConfig])
    def get_llm_configs(user_id: str = Depends(get_current_user_with_server)):
        """
        Retrieve the base configuration for the server.
        """
        interface.clear()
        return [server.server_llm_config]

    @router.get("/config/embedding", tags=["config"], response_model=List[EmbeddingConfig])
    def get_embedding_configs(user_id: str = Depends(get_current_user_with_server)):
        """
        Retrieve the base configuration for the server.
        """
        interface.clear()
        return [server.server_embedding_config]

    return router
