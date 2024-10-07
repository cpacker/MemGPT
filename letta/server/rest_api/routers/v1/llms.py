from typing import TYPE_CHECKING, List

from fastapi import APIRouter, Depends

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.server.rest_api.utils import get_letta_server

if TYPE_CHECKING:
    from letta.server.server import SyncServer

router = APIRouter(prefix="/models", tags=["models", "llms"])


@router.get("/", response_model=List[LLMConfig], operation_id="list_models")
def list_llm_backends(
    server: "SyncServer" = Depends(get_letta_server),
):

    models = server.list_llm_models()
    print(models)
    return models


@router.get("/embedding", response_model=List[EmbeddingConfig], operation_id="list_embedding_models")
def list_embedding_backends(
    server: "SyncServer" = Depends(get_letta_server),
):

    models = server.list_embedding_models()
    print(models)
    return models
