import uuid
from functools import partial
from typing import Optional

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field

from memgpt.data_types import User
from memgpt.models.pydantic_models import UserModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


# class ListHumansResponse(BaseModel):
#     humans: List[HumanModel] = Field(..., description="List of human configurations.")


class CreateUserRequest(BaseModel):
    id: uuid.UUID = Field(..., description="The user id.")
    default_agent: Optional[str] = Field(..., description="The user's default agent.")
    policies_accepted: bool = Field(..., description="Whether or not the users accepted policies.")


def setup_users_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/users/{user_id}", tags=["users"], response_model=UserModel)
    async def get_user(user_id: uuid.UUID):
        # Clear the interface
        interface.clear()
        user = server.get_user(user_id)
        if user is not None:
            return user
        return None

    @router.post("/users/create", tags=["users"], response_model=UserModel)
    async def create_user(
        request: CreateUserRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        new_user = User(id=request.id, default_agent=request.default_agent, policies_accepted=request.policies_accepted)
        server.create_user(new_user)

    return router
