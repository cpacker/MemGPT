from typing import TYPE_CHECKING, List

from fastapi import APIRouter, Depends

from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.llm_config import LLMConfig
from memgpt.server.rest_api.utils import get_memgpt_server

if TYPE_CHECKING:
    from memgpt.server.server import SyncServer

router = APIRouter(prefix="/models", tags=["models", "llms"])


@router.get("/", response_model=List[LLMConfig], operation_id="list_models")
def list_llm_backends(
    server: "SyncServer" = Depends(get_memgpt_server),
):

    return server.list_models()


@router.get("/embedding", response_model=List[EmbeddingConfig], operation_id="list_embedding_models")
def list_embedding_backends(
    server: "SyncServer" = Depends(get_memgpt_server),
):

    return server.list_embedding_models()
