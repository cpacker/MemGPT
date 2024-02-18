import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from memgpt.server.server import SyncServer

security = HTTPBearer()


def get_current_user(server: SyncServer, auth: HTTPAuthorizationCredentials = Depends(security)) -> uuid.UUID:
    try:
        api_key = auth.credentials
        user_id = server.api_key_to_user(api_key=api_key)
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {e}")
