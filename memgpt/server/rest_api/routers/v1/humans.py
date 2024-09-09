from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.block import Human
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.schemas.humans import CreateHumanRequest, ListHumansResponse

if TYPE_CHECKING:
    from memgpt.server.server import SyncServer


router = APIRouter(prefix="/humans", tags=["humans"])


@router.get("/", response_model=ListHumansResponse)
async def list_humans(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # Clear the interface

    humans = server.ms.list_humans(user_id=actor.id)
    return ListHumansResponse(humans=humans)


@router.post("/", response_model=Human)
async def create_human(
    request: CreateHumanRequest = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # TODO: disallow duplicate names for humans

    new_human = Human(text=request.text, name=request.name, user_id=actor.id)
    human_id = new_human.id
    server.ms.add_human(new_human)
    return Human(id=human_id, text=request.text, name=request.name, user_id=actor.id)


@router.delete("/{human_name}", response_model=Human)
async def delete_human(
    human_name: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    human = server.ms.delete_human(human_name, user_id=actor.id)
    return human


@router.get("/{human_name}", response_model=Human)
async def get_human(
    human_name: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    human = server.ms.get_human(human_name, user_id=actor.id)
    if human is None:
        raise HTTPException(status_code=404, detail="Human not found")
    return human
