from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from memgpt.schemas.api_key import APIKey, APIKeyCreate
from memgpt.schemas.user import User, UserCreate
from memgpt.server.rest_api.utils import get_memgpt_server

# from memgpt.server.schemas.users import (
#     CreateAPIKeyRequest,
#     CreateAPIKeyResponse,
#     CreateUserRequest,
#     CreateUserResponse,
#     DeleteAPIKeyResponse,
#     DeleteUserResponse,
#     GetAllUsersResponse,
#     GetAPIKeysResponse,
# )

if TYPE_CHECKING:
    from memgpt.schemas.user import User
    from memgpt.server.server import SyncServer


router = APIRouter(prefix="/users", tags=["users", "admin"])


@router.get("/", tags=["admin"], response_model=List[User], operation_id="list_users")
def get_all_users(
    cursor: Optional[str] = Query(None),
    limit: Optional[int] = Query(50),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get a list of all users in the database
    """
    try:
        next_cursor, users = server.ms.get_all_users(cursor=cursor, limit=limit)
        # processed_users = [{"user_id": user.id} for user in users]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return users


@router.post("/", tags=["admin"], response_model=User, operation_id="create_user")
def create_user(
    request: UserCreate = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Create a new user in the database
    """

    user = server.create_user(request)
    return user


@router.delete("/", tags=["admin"], response_model=User, operation_id="delete_user")
def delete_user(
    user_id: str = Query(..., description="The user_id key to be deleted."),
    server: "SyncServer" = Depends(get_memgpt_server),
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
    return user


@router.post("/keys", response_model=APIKey, operation_id="create_api_key")
def create_new_api_key(
    create_key: APIKeyCreate = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Create a new API key for a user
    """
    api_key = server.create_api_key(create_key)
    return api_key


@router.get("/keys", response_model=List[APIKey], operation_id="list_api_keys")
def get_api_keys(
    user_id: str = Query(..., description="The unique identifier of the user."),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get a list of all API keys for a user
    """
    if server.ms.get_user(user_id=user_id) is None:
        raise HTTPException(status_code=404, detail=f"User does not exist")
    api_keys = server.ms.get_all_api_keys_for_user(user_id=user_id)
    return api_keys


@router.delete("/keys", response_model=APIKey, operation_id="delete_api_key")
def delete_api_key(
    api_key: str = Query(..., description="The API key to be deleted."),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    return server.delete_api_key(api_key)
