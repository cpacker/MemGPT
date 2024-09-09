from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.tool import Tool, ToolCreate
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.server import SyncServer

router = APIRouter(prefix="/tools", tags=["tools"])


@router.delete("/{tool_id}", tags=["tools"])
def delete_tool(
    tool_id: str,
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Delete a tool by name
    """
    # actor = server.get_current_user()
    server.delete_tool(tool_id=tool_id)


@router.get("/{tool_id}", tags=["tools"], response_model=Tool)
def get_tool(
    tool_id: str,
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Get a tool by name
    """
    # actor = server.get_current_user()

    tool = server.get_tool(tool_id=tool_id)
    if tool is None:
        # return 404 error
        raise HTTPException(status_code=404, detail=f"Tool with id {tool_id} not found.")
    return tool


@router.get("/name/{tool_name}", tags=["tools"], response_model=str)
def get_tool_id(
    tool_name: str,
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Get a tool by name
    """
    actor = server.get_current_user()

    tool = server.get_tool_id(tool_name, user_id=actor.id)
    if tool is None:
        # return 404 error
        raise HTTPException(status_code=404, detail=f"Tool with name {tool_name} not found.")
    return tool


@router.get("/", tags=["tools"], response_model=List[Tool])
def list_all_tools(
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Get a list of all tools available to agents created by a user
    """
    server.get_current_user()

    # TODO: add back when user-specific
    # return server.list_tools(user_id=actor.id)
    return server.ms.list_tools(user_id=None)


@router.post("/", tags=["tools"], response_model=Tool)
def create_tool(
    tool: ToolCreate = Body(...),
    update: bool = False,
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Create a new tool
    """
    actor = server.get_current_user()

    return server.create_tool(
        request=tool,
        update=update,
        user_id=actor.id,
    )
