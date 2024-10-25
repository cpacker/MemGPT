from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException

from letta.orm.errors import NoResultFound
from letta.schemas.tool import Tool, ToolCreate, ToolUpdate
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

router = APIRouter(prefix="/tools", tags=["tools"])


@router.delete("/{tool_id}", operation_id="delete_tool")
def delete_tool(
    tool_id: str,
    server: SyncServer = Depends(get_letta_server),
):
    """
    Delete a tool by name
    """
    # actor = server.get_user_or_default(user_id=user_id)
    server.tool_manager.delete_tool(tool_id=tool_id)


@router.get("/{tool_id}", response_model=Tool, operation_id="get_tool")
def get_tool(
    tool_id: str,
    server: SyncServer = Depends(get_letta_server),
):
    """
    Get a tool by ID
    """
    tool = server.tool_manager.get_tool_by_id(tool_id=tool_id)
    if tool is None:
        # return 404 error
        raise HTTPException(status_code=404, detail=f"Tool with id {tool_id} not found.")
    return tool


@router.get("/name/{tool_name}", response_model=str, operation_id="get_tool_id_by_name")
def get_tool_id(
    tool_name: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Get a tool ID by name
    """
    actor = server.get_user_or_default(user_id=user_id)

    try:
        tool = server.tool_manager.get_tool_by_name_and_org_id(tool_name=tool_name, organization_id=actor.organization_id)
        return tool.id
    except NoResultFound:
        raise HTTPException(status_code=404, detail=f"Tool with name {tool_name} and organization id {actor.organization_id} not found.")


@router.get("/", response_model=List[Tool], operation_id="list_tools")
def list_tools(
    cursor: Optional[str] = None,
    limit: Optional[int] = 50,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Get a list of all tools available to agents belonging to the org of the user
    """
    try:
        actor = server.get_user_or_default(user_id=user_id)
        return server.tool_manager.list_tools_for_org(organization_id=actor.organization_id, cursor=cursor, limit=limit)
    except Exception as e:
        # Log or print the full exception here for debugging
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Tool, operation_id="create_tool")
def create_tool(
    request: ToolCreate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Create a new tool
    """
    # Derive user and org id from actor
    actor = server.get_user_or_default(user_id=user_id)
    request.organization_id = actor.organization_id
    request.user_id = actor.id

    # Send request to create the tool
    return server.tool_manager.create_or_update_tool(
        tool_create=request,
    )


@router.patch("/{tool_id}", response_model=Tool, operation_id="update_tool")
def update_tool(
    tool_id: str,
    request: ToolUpdate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Update an existing tool
    """
    return server.tool_manager.update_tool_by_id(tool_id, request)
