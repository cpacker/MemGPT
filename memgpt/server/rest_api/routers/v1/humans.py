from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, Body

from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.schemas.block import Human
from memgpt.server.schemas.humans import ListHumansResponse, CreateHumanRequest

if TYPE_CHECKING:
    from memgpt.server.server import SyncServer
    from memgpt.server.rest_api.interface import QueuingInterface


router = APIRouter(prefix="/humans", tags=["humans"])



@router.get("/",  response_model=ListHumansResponse)
async def list_humans(
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # Clear the interface
    interface.clear()
    humans = server.ms.list_humans(user_id=actor._id)
    return ListHumansResponse(humans=humans)

@router.post("/",  response_model=Human)
async def create_human(
    request: CreateHumanRequest = Body(...),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    # TODO: disallow duplicate names for humans
    interface.clear()
    new_human = Human(text=request.text, name=request.name, user_id=actor._id)
    human_id = new_human.id
    server.ms.add_human(new_human)
    return Human(id=human_id, text=request.text, name=request.name, user_id=actor._id)

@router.delete("/{human_name}",  response_model=Human)
async def delete_human(
    human_name: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    interface.clear()
    human = server.ms.delete_human(human_name, user_id=actor._id)
    return human

@router.get("/{human_name}",  response_model=Human)
async def get_human(
    human_name: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    actor = server.get_current_user()
    interface.clear()
    human = server.ms.get_human(human_name, user_id=actor._id)
    if human is None:
        raise HTTPException(status_code=404, detail="Human not found")
    return human