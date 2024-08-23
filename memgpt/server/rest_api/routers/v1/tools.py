from fastapi import APIRouter, Body, Depends, HTTPException
from typing import List

from memgpt.schemas.tool import Tool, ToolCreate
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.server.server import SyncServer

router = APIRouter(prefix="/tools", tags=["tools"])


@router.delete("/{tool_id}")
def delete_tool(
    tool_id: str,
    interface: QueuingInterface = Depends(get_current_interface),
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Delete a tool by name
    """
    actor = server.get_current_user()
    interface.clear()
    server.ms.delete_tool(id=tool_id, user_id=actor.id)


@router.get("/{tool_id}", tags=["tools"], response_model=Tool)
def get_tool(
    tool_id: str,
    interface: QueuingInterface = Depends(get_current_interface),
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Get a tool by name
    """
    actor = server.get_current_user()
    # Clear the interface
    interface.clear()
    if tool := server.ms.get_tool(tool_id=tool_id, user_id=actor.id):
        return tool
    # return 404 error
    # TODO issue #13 in the big spreadsheet: Standardize errors and correct error codes
    raise HTTPException(status_code=404, detail=f"Tool with id {tool_id} not found.")


@router.get("/", tags=["tools"], response_model=List[Tool])
def list_all_tools(
    interface: QueuingInterface = Depends(get_current_interface),
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Get a list of all tools available to agents created by a user
    """
    actor = server.get_current_user()
    # Clear the interface
    interface.clear()
    return server.ms.list_tools(user_id=actor.id)


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
