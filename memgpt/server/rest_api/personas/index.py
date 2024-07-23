import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
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

    @router.get("/personas/all", tags=["personas"], response_model=ListPersonasResponse)
    async def list_personas(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()

        personas = server.ms.list_personas(user_id=user_id)
        return ListPersonasResponse(personas=personas)

    @router.get("/personas/{name}", tags=["personas"], response_model=Optional[PersonaModel])
    async def get_persona(
        name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        persona = server.ms.get_persona(name, user_id)
        if persona is not None:
            return persona
        return None

    @router.post("/personas/add", tags=["personas"], response_model=PersonaModel)
    async def add_persona(
        request: CreatePersonaRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # TODO: disallow duplicate names for personas
        interface.clear()
        new_persona = PersonaModel(text=request.text, name=request.name, user_id=user_id)
        persona_id = new_persona.id
        server.ms.add_persona(new_persona)
        return PersonaModel(id=persona_id, text=request.text, name=request.name, user_id=user_id)

    @router.post("/personas/update", tags=["personas"], response_model=PersonaModel)
    async def update_persona(
        request: CreatePersonaRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        original_persona = server.ms.get_persona(request.name, user_id)
        assert original_persona, f"Persona name {request.name} or {user_id} should already exist but does not."
        original_persona.text = request.text
        server.ms.update_persona(original_persona)
        return PersonaModel(id=original_persona.id, text=request.text, name=request.name, user_id=user_id)

    @router.delete("/personas/{persona_name}", tags=["personas"], response_model=PersonaModel)
    async def delete_persona(
        persona_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        persona = server.ms.delete_persona(persona_name, user_id=user_id)
        return persona

    @router.get("/personas/{persona_name}", tags=["personas"], response_model=PersonaModel)
    async def get_persona(
        persona_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        persona = server.ms.get_persona(persona_name, user_id=user_id)
        if persona is None:
            raise HTTPException(status_code=404, detail="Persona not found")
        return persona

    return router
