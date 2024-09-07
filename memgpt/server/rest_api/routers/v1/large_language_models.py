from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from memgpt.schemas.llm_config import LLMConfig
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.schemas.large_language_models import ListModelsResponse

if TYPE_CHECKING:
    from memgpt.server.server import SyncServer

router = APIRouter(prefix="/models", tags=["models", "large_language_models"])


@router.get("/", response_model=ListModelsResponse)
def list_models(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    server.get_current_user()

    # currently, the server only supports one model, however this may change in the future
    llm_config = LLMConfig(
        model=server.server_llm_config.model,
        model_endpoint=server.server_llm_config.model_endpoint,
        model_endpoint_type=server.server_llm_config.model_endpoint_type,
        model_wrapper=server.server_llm_config.model_wrapper,
        context_window=server.server_llm_config.context_window,
    )

    return ListModelsResponse(models=[llm_config])
