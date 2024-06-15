from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from memgpt.log import get_logger
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

logger = get_logger(__name__)
router = APIRouter()


class AuthResponse(BaseModel):
    uuid: UUID = Field(..., description="UUID of the user")


class AuthRequest(BaseModel):
    password: str = Field(None, description="Admin password provided when starting the MemGPT server")


def setup_auth_router(server: SyncServer, interface: QueuingInterface, password: str) -> APIRouter:

    @router.post("/auth", tags=["auth"], response_model=AuthResponse)
    def authenticate_user(request: AuthRequest) -> AuthResponse:
        """
        Authenticates the user and sends response with User related data.

        Currently, this is a placeholder that simply returns a UUID placeholder
        """
        interface.clear()
        if request.password != password:
            response = server.api_key_to_user(api_key=request.password)
        else:
            response = server.authenticate_user()
        return AuthResponse(uuid=response)

    return router
