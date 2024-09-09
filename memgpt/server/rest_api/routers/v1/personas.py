from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from memgpt.schemas.block import Persona
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.schemas.personas import CreatePersonaRequest, ListPersonasResponse

if TYPE_CHECKING:
    from memgpt.server.server import SyncServer


router = APIRouter(prefix="/personas", tags=["personas"])


@router.get("/", response_model=ListPersonasResponse)
async def list_personas(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    personas = server.ms.list_personas(user_id=actor.id)
    return ListPersonasResponse(personas=personas)


@router.post("/", response_model=Persona)
async def create_persona(
    persona: CreatePersonaRequest,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # TODO: disallow duplicate names for personas

    new_persona = Persona(text=persona.text, name=persona.name, user_id=actor.id)
    persona_id = new_persona.id
    server.ms.add_persona(new_persona)
    return Persona(id=persona_id, text=persona.text, name=persona.name, user_id=actor.id)


@router.delete("/{persona_name}", response_model=Persona)
async def delete_persona(
    persona_name: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    persona = server.ms.delete_persona(persona_name, user_id=actor.id)
    return persona


@router.get("/{persona_name}", response_model=Persona)
async def get_persona(
    persona_name: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()

    persona = server.ms.get_persona(persona_name, user_id=actor.id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona
