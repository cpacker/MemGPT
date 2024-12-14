from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from letta.schemas.user import User, UserCreate, UserUpdate
from letta.server.rest_api.utils import get_letta_server

if TYPE_CHECKING:
    from letta.schemas.user import User
    from letta.server.server import SyncServer


router = APIRouter(prefix="/users", tags=["users", "admin"])


@router.get("/", tags=["admin"], response_model=List[User], operation_id="list_users")
def list_users(
    cursor: Optional[str] = Query(None),
    limit: Optional[int] = Query(50),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Get a list of all users in the database
    """
    try:
        next_cursor, users = server.user_manager.list_users(cursor=cursor, limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return users


@router.post("/", tags=["admin"], response_model=User, operation_id="create_user")
def create_user(
    request: UserCreate = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Create a new user in the database
    """
    user = User(**request.model_dump())
    user = server.user_manager.create_user(user)
    return user


@router.put("/", tags=["admin"], response_model=User, operation_id="update_user")
def update_user(
    user: UserUpdate = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Update a user in the database
    """
    user = server.user_manager.update_user(user)
    return user


@router.delete("/", tags=["admin"], response_model=User, operation_id="delete_user")
def delete_user(
    user_id: str = Query(..., description="The user_id key to be deleted."),
    server: "SyncServer" = Depends(get_letta_server),
):
    # TODO make a soft deletion, instead of a hard deletion
    try:
        user = server.user_manager.get_user_by_id(user_id=user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User does not exist")
        server.user_manager.delete_user_by_id(user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return user
