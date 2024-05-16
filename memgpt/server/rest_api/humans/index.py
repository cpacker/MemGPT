import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends
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

    @router.get("/humans", tags=["humans"], response_model=ListHumansResponse)
    async def list_humans(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        humans = server.ms.list_humans(user_id=user_id)
        return ListHumansResponse(humans=humans)

    @router.post("/humans", tags=["humans"], response_model=HumanModel)
    async def create_human(
        request: CreateHumanRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # TODO: disallow duplicate names for humans
        interface.clear()
        new_human = HumanModel(text=request.text, name=request.name, user_id=user_id)
        human_id = new_human.id
        server.ms.add_human(new_human)
        return HumanModel(id=human_id, text=request.text, name=request.name, user_id=user_id)

    return router
