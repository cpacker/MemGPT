
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException

from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.server.schemas.personas import ListPersonasResponse, CreatePersonaRequest
from memgpt.schemas.block import Persona

if TYPE_CHECKING:
    from memgpt.schemas.user import User
    from memgpt.server.server import SyncServer
    from memgpt.server.rest_api.interface import QueuingInterface


router = APIRouter(prefix="/personas", tags=["personas"])


@router.get("/", response_model=ListPersonasResponse)
async def list_personas(
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # Clear the interface
    interface.clear()

    personas = server.ms.list_personas(user_id=actor.id)
    return ListPersonasResponse(personas=personas)

@router.post("/", response_model=Persona)
async def create_persona(
    persona: CreatePersonaRequest,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # TODO: disallow duplicate names for personas
    interface.clear()
    new_persona = Persona(text=persona.text, name=persona.name, user_id=actor.id)
    persona_id = new_persona.id
    server.ms.add_persona(new_persona)
    return Persona(id=persona_id, text=persona.text, name=persona.name, user_id=actor.id)

@router.delete("/{persona_name}", response_model=Persona)
async def delete_persona(
    persona_name: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    interface.clear()
    persona = server.ms.delete_persona(persona_name, user_id=actor.id)
    return persona

@router.get("/{persona_name}", response_model=Persona)
async def get_persona(
    persona_name: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    interface.clear()
    persona = server.ms.get_persona(persona_name, user_id=actor.id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona
