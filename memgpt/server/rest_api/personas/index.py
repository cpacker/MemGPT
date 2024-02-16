import uuid
from typing import List
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel, Field
from functools import partial

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.auth_token import get_current_user

router = APIRouter()


class ListPersonasResponse(BaseModel):
    personas: List[dict] = Field(..., description="List of persona configurations.")


def setup_personas_index_router(server: SyncServer, interface: QueuingInterface):
    get_current_user_with_server = partial(get_current_user, server)

    @router.get("/personas", tags=["personas"], response_model=ListPersonasResponse)
    async def list_personas(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
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
