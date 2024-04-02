from functools import partial
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

from memgpt.models.pydantic_models import LLMConfigModel, EmbeddingConfigModel

router = APIRouter()


class ListModelsResponse(BaseModel):
    models: List[LLMConfigModel] = Field(..., description="List of model configurations.")


def setup_models_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    partial(partial(get_current_user, server), password)

    @router.get("/models", tags=["models"], response_model=ListModelsResponse)
    async def list_models():
        # Clear the interface
        interface.clear()

        # currently, the server only supports one model, however this may change in the future
        llm_config = LLMConfigModel(
            model=server.server_llm_config.model,
            model_endpoint=server.server_llm_config.model_endpoint,
            model_endpoint_type=server.server_llm_config.model_endpoint_type,
            model_wrapper=server.server_llm_config.model_wrapper,
            context_window=server.server_llm_config.context_window,
        )

        return ListModelsResponse(models=[llm_config])

    return router
