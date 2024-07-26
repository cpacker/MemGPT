from typing import TYPE_CHECKING, List
from fastapi import APIRouter, Depends

from memgpt.server.rest_api.utils import get_memgpt_server, get_current_interface
from memgpt.models.pydantic_models import LLMConfigModel
from memgpt.server.schemas.large_language_models import ListModelsResponse

if TYPE_CHECKING:
    from memgpt.server.server import SyncServer
    from memgpt.orm.user import User
    from memgpt.server.rest_api.interface import QueuingInterface

router = APIRouter(prefix="/models", tags=["models","large_language_models"])

@router.get("/", response_model=ListModelsResponse)
def list_models(
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),

):
    actor = server.get_current_user()
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
