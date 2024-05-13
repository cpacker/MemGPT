import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import PersonaModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ListPersonasResponse(BaseModel):
    personas: List[PersonaModel] = Field(..., description="List of persona configurations.")


class CreatePersonaRequest(BaseModel):
    text: str = Field(..., description="The persona text.")
    name: str = Field(..., description="The name of the persona.")


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

    @router.post("/personas", tags=["personas"], response_model=PersonaModel)
    async def create_persona(
        request: CreatePersonaRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # TODO: disallow duplicate names for personas
        interface.clear()
        new_persona = PersonaModel(text=request.text, name=request.name, user_id=user_id)
        persona_id = new_persona.id
        server.ms.add_persona(new_persona)
        return PersonaModel(id=persona_id, text=request.text, name=request.name, user_id=user_id)

    return router
