import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.models.pydantic_models import PersonaModel

router = APIRouter()


class ListPersonasResponse(BaseModel):
    personas: List[PersonaModel] = Field(..., description="List of persona configurations.")


def setup_personas_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/personas", tags=["personas"], response_model=ListPersonasResponse)
    async def list_personas(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()

        personas = server.ms.list_personas(user_id=user_id)
        return ListPersonasResponse(personas=personas)

    return router
