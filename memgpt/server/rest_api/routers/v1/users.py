from typing import TYPE_CHECKING, Optional
from fastapi import APIRouter, Depends, HTTPException, Body, Query

from memgpt.server.rest_api.utils import get_current_user, get_current_interface, get_memgpt_server
from memgpt.server.schemas.users import (
    CreateUserRequest,
    CreateUserResponse,
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    GetAPIKeysResponse,
    DeleteAPIKeyResponse,
    DeleteUserResponse,
    GetAllUsersResponse,
)

if TYPE_CHECKING:
    from uuid import UUID
    from memgpt.models.pydantic_models import User
    from memgpt.server.server import SyncServer
    from memgpt.server.rest_api.interface import QueuingInterface


router = APIRouter(prefix="/users", tags=["users","admin"])





@router.get("/users", tags=["admin"], response_model=GetAllUsersResponse)
def get_all_users(cursor: Optional["UUID"] = Query(None), limit: Optional[int] = Query(50)):
    """
    Get a list of all users in the database
    """
    try:
        next_cursor, users = server.ms.get_all_users(cursor, limit)
        processed_users = [{"user_id": user.id} for user in users]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return GetAllUsersResponse(cursor=next_cursor, user_list=processed_users)

@router.post("/users", tags=["admin"], response_model=CreateUserResponse)
def create_user(request: Optional[CreateUserRequest] = Body(None)):
    """
    Create a new user in the database
    """
    if request is None:
        request = CreateUserRequest()

    new_user = User(
        id=None if not request.user_id else request.user_id,
        # TODO can add more fields (name? metadata?)
    )

    try:
        server.ms.create_user(new_user)

        # make sure we can retrieve the user from the DB too
        new_user_ret = server.ms.get_user(new_user.id)
        if new_user_ret is None:
            raise HTTPException(status_code=500, detail=f"Failed to verify user creation")

        # create an API key for the user
        token = server.ms.create_api_key(user_id=new_user.id, name=request.api_key_name)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return CreateUserResponse(user_id=new_user_ret.id, api_key=token.token)

@router.delete("/users", tags=["admin"], response_model=DeleteUserResponse)
def delete_user(
    user_id: "UUID" = Query(..., description="The user_id key to be deleted."),
):
    # TODO make a soft deletion, instead of a hard deletion
    try:
        user = server.ms.get_user(user_id=user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User does not exist")
        server.ms.delete_user(user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return DeleteUserResponse(message="User successfully deleted.", user_id_deleted=user_id)

@router.post("/keys", response_model=CreateAPIKeyResponse)
def create_new_api_key(
    create_key: CreateAPIKeyRequest = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),

):
    """
    Create a new API key for a user
    """
    token = server.ms.create_api_key(user_id=create_key.user_id, name=create_key.name)
    return CreateAPIKeyResponse(api_key=token.token)

@router.get("/keys", response_model=GetAPIKeysResponse)
def get_api_keys(
    user_id: "UUID" = Query(..., description="The unique identifier of the user."),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get a list of all API keys for a user
    """
    if server.ms.get_user(user_id=user_id) is None:
        raise HTTPException(status_code=404, detail=f"User does not exist")
    tokens = server.ms.get_all_api_keys_for_user(user_id=user_id)
    processed_tokens = [t.token for t in tokens]
    return GetAPIKeysResponse(api_key_list=processed_tokens)

@router.delete("/keys", response_model=DeleteAPIKeyResponse)
def delete_api_key(
    api_key: str = Query(..., description="The API key to be deleted."),
    actor: "User" = Depends(get_current_user),
    server: "SyncServer" = Depends(get_memgpt_server),

):
    token = server.ms.get_api_key(api_key=api_key)
    if token is None:
        raise HTTPException(status_code=404, detail=f"API key does not exist")
    server.ms.delete_api_key(api_key=api_key)
    return DeleteAPIKeyResponse(message="API key successfully deleted.", api_key_deleted=api_key)
