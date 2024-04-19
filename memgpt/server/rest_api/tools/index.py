import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import ToolModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ListToolsResponse(BaseModel):
    tools: List[ToolModel] = Field(..., description="List of tools (functions).")


def setup_tools_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/tools", tags=["tools"], response_model=ListToolsResponse)
    async def list_tools(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Get a list of all tools available to agents created by a user
        """
        # Clear the interface
        interface.clear()
        tools = server.ms.list_tools(user_id=user_id)
        return ListToolsResponse(tools=tools)

    return router
