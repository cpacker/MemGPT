from typing import TYPE_CHECKING
from fastapi import APIRouter, Body, Depends, HTTPException
from memgpt.server.schemas.tools import CreateToolRequest, ListToolsResponse, ToolModel
from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

if TYPE_CHECKING:
    from memgpt.orm.user import User

router = APIRouter(prefix="/tools", tags=["tools"])

@router.delete("/{tool_name}")
def delete_tool(
    tool_name: str,
    interface: QueuingInterface = Depends(get_current_interface),
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    Delete a tool by name
    """
    actor = server.get_current_user()
    interface.clear()
    server.ms.delete_tool(name=tool_name, user_id=actor._id)

@router.get("/tools/{tool_name}", tags=["tools"], response_model=ToolModel)
def get_tool(
    tool_name: str,
    interface: QueuingInterface = Depends(get_current_interface),
    server: SyncServer = Depends(get_memgpt_server),
    ):
    """
    Get a tool by name
    """
    actor = server.get_current_user()
    # Clear the interface
    interface.clear()
    if tool := server.ms.get_tool(tool_name=tool_name, user_id=actor._id):
        return tool
    # return 404 error
    # TODO issue #13 in the big spreadsheet: Standardize errors and correct error codes
    raise HTTPException(status_code=404, detail=f"Tool with name {tool_name} not found.")

@router.get("/tools", tags=["tools"])
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

@router.post("/tools", tags=["tools"], response_model=ToolModel)
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
            json_schema=tool.json_schema,
            source_code=tool.source_code,
            source_type=tool.source_type,
            tags=tool.tags,
            user_id=actor._id,
            exists_ok=tool.update,
        )
    # TODO issue #14 - code quality
    # TDOD issue #13 - Standardize errors and correct error codes
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Failed to create tool: {e}, exists_ok={tool.update}")


