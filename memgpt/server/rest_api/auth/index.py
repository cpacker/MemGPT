from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from memgpt.log import get_logger
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

logger = get_logger(__name__)
router = APIRouter()


# TODO: remove these and add them to schemas?
class AuthResponse(BaseModel):
    user_id: str = Field(..., description="ID of the user")
    is_admin: Optional[bool] = Field(None, description="Whether the user is an admin")


class AuthRequest(BaseModel):
    password: str = Field(None, description="Admin password provided when starting the MemGPT server")


def setup_auth_router(server: SyncServer, interface: QueuingInterface, password: str) -> APIRouter:

    @router.post("/auth", tags=["auth"], response_model=AuthResponse)
    def authenticate_user(request: AuthRequest) -> AuthResponse:
        """
        Authenticates the user and sends response with User related data.

        """
        interface.clear()

        is_admin = False
        if request.password != password:
            user_id = server.api_key_to_user(api_key=request.password)
            return user_id
        else:
            is_admin = True
            user_id = server.authenticate_user()
            return None
        return AuthResponse(user_id=user_id, is_admin=is_admin)

    return router
