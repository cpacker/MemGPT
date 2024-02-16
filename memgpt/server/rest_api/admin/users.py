import uuid
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from memgpt.data_types import User
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.auth_token import get_current_user

router = APIRouter()


class GetAllUsersResponse(BaseModel):
    user_list: List[dict] = Field(..., description="A list of users.")


class CreateUserRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Identifier of the user (optional, generated automatically if null).")


class CreateUserResponse(BaseModel):
    user_id: str = Field(..., description="Identifier of the user (UUID).")


class CreateAPIKeyRequest(BaseModel):
    user_id: str = Field(..., description="Identifier of the user (UUID).")
    name: Optional[str] = Field(None, description="Name for the API key (optional).")


class CreateAPIKeyResponse(BaseModel):
    api_key: str = Field(..., description="New API key generated.")


class GetAPIKeysRequest(BaseModel):
    user_id: str = Field(..., description="Identifier of the user (UUID).")


class GetAPIKeysResponse(BaseModel):
    api_key_list: List[str] = Field(..., description="Identifier of the user (UUID).")


class DeleteAPIKeyResponse(BaseModel):
    message: str
    api_key_deleted: str


def setup_admin_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/users", tags=["users"], response_model=GetAllUsersResponse)
    def get_all_users():
        """
        Get a list of all users in the database
        """
        try:
            users = server.ms.get_all_users()
            processed_users = [{"user_id": user.id} for user in users]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return GetAllUsersResponse(user_list=processed_users)

    @router.post("/users", tags=["users"], response_model=CreateUserResponse)
    def create_user(request: CreateUserRequest = Body(...)):
        """
        Create a new user in the database
        """
        new_user = User(
            id=uuid.UUID(request.user_id) if request.user_id is not None else None,
            # TODO can add more fields (name? metadata?)
        )

        try:
            server.ms.create_user(new_user)
            # make sure we can retrieve the user from the DB too
            new_user_ret = server.ms.get_user(new_user.id)
            if new_user_ret is None:
                raise HTTPException(status_code=500, detail=f"Failed to verify user creation")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return CreateUserResponse(user_id=new_user_ret.id)

    # TODO add delete_user route

    @router.post("/users/keys", tags=["users"], response_model=CreateAPIKeyResponse)
    def create_new_api_key(request: CreateAPIKeyRequest = Body(...)):
        """
        Create a new API key for a user
        """
        try:
            user_id = uuid.UUID(request.user_id)
            token = server.ms.create_api_key(user_id=user_id, name=request.name)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return CreateAPIKeyResponse(api_key=token.token)

    @router.get("/users/keys", tags=["users"], response_model=GetAPIKeysResponse)
    def get_api_keys(
        user_id: str = Query(..., description="The unique identifier of the user."),
    ):
        """
        Get a list of all API keys for a user
        """
        try:
            user_id = uuid.UUID(user_id)
            tokens = server.ms.get_all_api_keys_for_user(user_id=user_id)
            processed_tokens = [t.token for t in tokens]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return CreateAPIKeyResponse(api_key=processed_tokens)

    @router.delete("/users/keys", tags=["users"], response_model=DeleteAPIKeyResponse)
    def delete_api_key(
        api_key: str = Query(..., description="The API key to be deleted."),
    ):
        try:
            token = server.ms.get_api_key(api_key=api_key)
            if token is None:
                raise HTTPException(status_code=404, detail=f"API key does not exist")
            server.ms.delete_api_key(api_key=api_key)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return DeleteAPIKeyResponse(message="API key successfully deleted.", api_key_deleted=api_key)

    return router
