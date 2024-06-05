from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter(prefix="/auth")


class AuthResponse(BaseModel):
    uuid: UUID = Field(..., description="UUID of the user")


class AuthRequest(BaseModel):
    password: str = Field(None, description="Admin password provided when starting the MemGPT server")


def setup_auth_router(server: SyncServer, interface: QueuingInterface, password: str) -> APIRouter:

    @router.post("/", tags=["auth"], response_model=AuthResponse)
    def authenticate_user(request: AuthRequest) -> AuthResponse:
        """
        Authenticates the user and sends response with User related data.

        Currently, this is a placeholder that simply returns a UUID placeholder
        """
        interface.clear()
        try:
            if request.password != password:
                # raise HTTPException(status_code=400, detail="Incorrect credentials")
                response = server.api_key_to_user(api_key=request.password)
            else:
                response = server.authenticate_user()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return AuthResponse(uuid=response)

    @router.post("/jwt-auth", tags=["auth"])
    def login_for_access_token():
        """
        converts the given API key to a jwt refresh token
        """
        # decode and validate api token
        # get user
        # create refresh token for user
        # return refresh token

    @router.post("/jwt-refresh", tags=["auth"])
    def refresh_access_token(
        # scopes? how do we want to scope this?
        # I think for now this is hard-keyed, something like:
        # {"admin": bool, "agentId": str } - because a jwt should be scoped to an agent right?
        # maybe also this is when we introduce an idea of "team" or "organization" - this is missing right now.

    ):
        """
        creates a new jwt access token using the given refresh token
        """
        # decode and validate refresh token
        # create access token for user
        # return access token

    return router
