from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from letta.schemas.api_key import APIKey, APIKeyCreate
from letta.schemas.user import User, UserCreate
from letta.server.rest_api.interface import QueuingInterface
from letta.server.server import SyncServer

router = APIRouter()


def setup_admin_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/users", tags=["admin"], response_model=List[User])
    def get_all_users(cursor: Optional[str] = Query(None), limit: Optional[int] = Query(50)):
        """
        Get a list of all users in the database
        """
        try:
            # TODO: make this call a server function
            _, users = server.ms.get_all_users(cursor=cursor, limit=limit)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return users

    @router.post("/users", tags=["admin"], response_model=User)
    def create_user(request: UserCreate = Body(...)):
        """
        Create a new user in the database
        """
        try:
            user = server.user_manager.create_user(request)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return user

    @router.delete("/users", tags=["admin"], response_model=User)
    def delete_user(
        user_id: str = Query(..., description="The user_id key to be deleted."),
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

    @router.post("/users/keys", tags=["admin"], response_model=APIKey)
    def create_new_api_key(request: APIKeyCreate = Body(...)):
        """
        Create a new API key for a user
        """
        try:
            api_key = server.create_api_key(request)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return api_key

    @router.get("/users/keys", tags=["admin"], response_model=List[APIKey])
    def get_api_keys(
        user_id: str = Query(..., description="The unique identifier of the user."),
    ):
        """
        Get a list of all API keys for a user
        """
        try:
            if server.ms.get_user(user_id=user_id) is None:
                raise HTTPException(status_code=404, detail=f"User does not exist")
            api_keys = server.ms.get_all_api_keys_for_user(user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return api_keys

    @router.delete("/users/keys", tags=["admin"], response_model=APIKey)
    def delete_api_key(
        api_key: str = Query(..., description="The API key to be deleted."),
    ):
        try:
            return server.delete_api_key(api_key)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
