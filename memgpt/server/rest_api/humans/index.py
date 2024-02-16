import uuid
from typing import List
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel, Field
from functools import partial

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.auth_token import get_current_user

router = APIRouter()


class ListHumansResponse(BaseModel):
    humans: List[dict] = Field(..., description="List of human configurations.")


def setup_humans_index_router(server: SyncServer, interface: QueuingInterface):
    get_current_user_with_server = partial(get_current_user, server)

    @router.get("/humans", tags=["humans"], response_model=ListHumansResponse)
    async def list_humans(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()

        # TODO: Replace with actual data fetching logic once available
        humans_data = [
            {"name": "Marco", "text": "About Me"},
            {"name": "Sam", "text": "About Me 2"},
            {"name": "Bruce", "text": "About Me 3"},
        ]

        return ListHumansResponse(humans=humans_data)

    return router
