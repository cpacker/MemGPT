from fastapi import APIRouter, Depends

from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.schemas.config import ConfigResponse

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally
from memgpt.server.server import SyncServer

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/", response_model=ConfigResponse)
def get_server_config(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Retrieve the base configuration for the server.
    """

    response = server.get_server_config(include_defaults=True)
    return ConfigResponse(config=response["config"], defaults=response["defaults"])
