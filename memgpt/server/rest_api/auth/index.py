from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class AuthResponse(BaseModel):
    uuid: UUID = Field(..., description="UUID of the user")


def setup_auth_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/auth", tags=["auth"], response_model=AuthResponse)
    def authenticate_user():
        """
        Authenticates the user and sends response with User related data.

        Currently, this is a placeholder that simply returns a UUID placeholder
        """
        interface.clear()
        try:
            response = server.authenticate_user()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return AuthResponse(uuid=response)

    return router
