import uuid
from typing import List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface

router = APIRouter()


class ListHumansResponse(BaseModel):
    humans: List[dict] = Field(..., description="List of human configurations.")


def setup_humans_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/humans", tags=["humans"], response_model=ListHumansResponse)
    async def list_humans(user_id: str = Query(..., description="Unique identifier of the user.")):
        # Validate and parse the user ID
        user_id = None if user_id == "null" else user_id
        user_id = uuid.UUID(user_id) if user_id else None

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
