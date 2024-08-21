from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.tool import Tool, ToolCreate
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.server.schemas.tools import CreateToolRequest, ListToolsResponse
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
    server.ms.delete_tool(id=tool_id, user_id=actor._id)


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
    if tool := server.ms.get_tool(tool_id=tool_id, user_id=actor._id):
        return tool
    # return 404 error
    # TODO issue #13 in the big spreadsheet: Standardize errors and correct error codes
    raise HTTPException(status_code=404, detail=f"Tool with id {tool_id} not found.")


@router.get("/", tags=["tools"])
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
    return ListToolsResponse(tools=server.ms.list_tools(user_id=actor._id))


@router.post("/", tags=["tools"], response_model=Tool)
def create_tool(
    tool: CreateToolRequest = Body(...),
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Create a new tool
    """
    actor = server.get_current_user()
    try:
        return server.create_tool(
            ToolCreate(json_schema=tool.json_schema, source_code=tool.source_code, source_type=tool.source_type, tags=tool.tags),
            user_id=actor._id,
        )
    # TODO issue #14 - code quality
    # TDOD issue #13 - Standardize errors and correct error codes
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Failed to create tool: {e}, exists_ok={tool.update}")
