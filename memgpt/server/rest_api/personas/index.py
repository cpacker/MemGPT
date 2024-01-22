import uuid
from typing import List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface

router = APIRouter()


class ListPersonasResponse(BaseModel):
    personas: List[dict] = Field(..., description="List of persona configurations.")


def setup_personas_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/personas", tags=["personas"], response_model=ListPersonasResponse)
    async def list_personas(user_id: str = Query(..., description="Unique identifier of the user.")):
        # Validate and parse the user ID
        user_id = None if user_id == "null" else user_id
        user_id = uuid.UUID(user_id) if user_id else None

        # Clear the interface
        interface.clear()

        # TODO: Replace with actual data fetching logic once available
        personas_data = [
            {"name": "Persona 1", "text": "Details about Persona 1"},
            {"name": "Persona 2", "text": "Details about Persona 2"},
            {"name": "Persona 3", "text": "Details about Persona 3"},
        ]

        return ListPersonasResponse(personas=personas_data)

    return router
