from fastapi import APIRouter, Depends

from memgpt.server.rest_api.utils import get_current_interface,  get_memgpt_server
from memgpt.server.schemas.config import ConfigResponse

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally
from uuid import UUID
from memgpt.orm.user import User
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface


router = APIRouter(prefix="/config", tags=["config"])

@router.get("/", response_model=ConfigResponse)
def get_server_config(
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Retrieve the base configuration for the server.
    """
    interface.clear()
    response = server.get_server_config(include_defaults=True)
    return ConfigResponse(config=response["config"], defaults=response["defaults"])
