import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import HumanModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ListHumansResponse(BaseModel):
    humans: List[HumanModel] = Field(..., description="List of human configurations.")


class CreateHumanRequest(BaseModel):
    text: str = Field(..., description="The human text.")
    name: str = Field(..., description="The name of the human.")


def setup_humans_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/humans/all", tags=["humans"], response_model=ListHumansResponse)
    async def list_humans(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        humans = server.ms.list_humans(user_id=user_id)
        return ListHumansResponse(humans=humans)

    @router.get("/humans/{name}", tags=["humans"], response_model=Optional[HumanModel])
    async def get_human(
        name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        human = server.ms.get_human(name, user_id)
        if human is not None:
            return human
        return None

    @router.post("/humans/add", tags=["humans"], response_model=HumanModel)
    async def add_human(
        request: CreateHumanRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # TODO: disallow duplicate names for humans
        interface.clear()
        new_human = HumanModel(text=request.text, name=request.name, user_id=user_id)
        human_id = new_human.id
        server.ms.add_human(new_human)
        return HumanModel(id=human_id, text=request.text, name=request.name, user_id=user_id)

    @router.post("/humans/update", tags=["humans"], response_model=HumanModel)
    async def update_human(
        request: CreateHumanRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # TODO: disallow duplicate names for humans
        interface.clear()
        original_human = server.ms.get_human(request.name, user_id)
        assert original_human, f"Human {request.name} or {user_id} should already exist but does not."
        original_human.text = request.text
        server.ms.update_human(original_human)
        return HumanModel(id=original_human.id, text=request.text, name=request.name, user_id=user_id)

    @router.delete("/humans/{human_name}", tags=["humans"], response_model=HumanModel)
    async def delete_human(
        human_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        human = server.ms.delete_human(human_name, user_id=user_id)
        return human

    @router.get("/humans/{human_name}", tags=["humans"], response_model=HumanModel)
    async def get_human(
        human_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        human = server.ms.get_human(human_name, user_id=user_id)
        if human is None:
            raise HTTPException(status_code=404, detail="Human not found")
        return human

    return router
