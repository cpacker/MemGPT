
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException

from memgpt.server.rest_api.utils import get_current_user, get_current_interface, get_memgpt_server
from memgpt.server.schemas.personas import ListPersonasResponse, CreatePersonaRequest, PersonaModel

if TYPE_CHECKING:
    from memgpt.models.pydantic_models import User
    from memgpt.server.server import SyncServer
    from memgpt.server.rest_api.interface import QueuingInterface


router = APIRouter(prefocix="/personas", tags=["personas"])


@router.get("/", response_model=ListPersonasResponse)
async def list_personas(
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    # Clear the interface
    interface.clear()

    personas = server.ms.list_personas(user_id=actor._id)
    return ListPersonasResponse(personas=personas)

@router.post("/", response_model=PersonaModel)
async def create_persona(
    persona: CreatePersonaRequest,
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    # TODO: disallow duplicate names for personas
    interface.clear()
    new_persona = PersonaModel(text=persona.text, name=persona.name, user_id=actor._id)
    persona_id = new_persona.id
    server.ms.add_persona(new_persona)
    return PersonaModel(id=persona_id, text=persona.text, name=persona.name, user_id=actor._id)

@router.delete("/{persona_name}", response_model=PersonaModel)
async def delete_persona(
    persona_name: str,
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    interface.clear()
    persona = server.ms.delete_persona(persona_name, user_id=actor._id)
    return persona

@router.get("/{persona_name}", response_model=PersonaModel)
async def get_persona(
    persona_name: str,
    actor: "User" = Depends(get_current_user),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    interface.clear()
    persona = server.ms.get_persona(persona_name, user_id=actor._id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona
