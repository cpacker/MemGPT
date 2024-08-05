from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.tool import Tool, ToolCreate
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_user_tools_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.delete("/tools/{tool_id}", tags=["tools"])
    async def delete_tool(
        tool_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Delete a tool by name
        """
        # Clear the interface
        interface.clear()
        server.delete_tool(id)

    @router.get("/tools/{tool_id}", tags=["tools"], response_model=Tool)
    async def get_tool(
        tool_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get a tool by name
        """
        # Clear the interface
        interface.clear()
        tool = server.get_tool(tool_id)
        if tool is None:
            # return 404 error
            raise HTTPException(status_code=404, detail=f"Tool with id {tool_id} not found.")
        return tool

    @router.get("/tools", tags=["tools"], response_model=List[Tool])
    async def list_all_tools(
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get a list of all tools available to agents created by a user
        """
        # Clear the interface
        interface.clear()
        return server.list_tools(user_id)

    @router.post("/tools", tags=["tools"], response_model=Tool)
    async def create_tool(
        request: ToolCreate = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Create a new tool
        """
        try:
            return server.create_tool(request, user_id=user_id)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=f"Failed to create tool: {e}")

    return router
