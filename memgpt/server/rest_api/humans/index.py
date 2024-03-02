import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.models.pydantic_models import HumanModel

router = APIRouter()


class ListHumansResponse(BaseModel):
    humans: List[HumanModel] = Field(..., description="List of human configurations.")


def setup_humans_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/humans", tags=["humans"], response_model=ListHumansResponse)
    async def list_humans(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        humans = server.ms.list_humans(user_id=user_id)
        return ListHumansResponse(humans=humans)

    return router
