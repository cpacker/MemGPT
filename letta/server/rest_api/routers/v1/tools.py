from typing import List, Optional

from composio.client.collections import ActionModel, AppModel
from fastapi import APIRouter, Body, Depends, Header, HTTPException

from letta.errors import LettaToolCreateError
from letta.orm.errors import UniqueConstraintViolationError
from letta.schemas.letta_message import FunctionReturn
from letta.schemas.tool import Tool, ToolCreate, ToolRun, ToolRunFromSource, ToolUpdate
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

router = APIRouter(prefix="/tools", tags=["tools"])


@router.delete("/{tool_id}", operation_id="delete_tool")
def delete_tool(
    tool_id: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Delete a tool by name
    """
    actor = server.get_user_or_default(user_id=user_id)
    server.tool_manager.delete_tool_by_id(tool_id=tool_id, actor=actor)


@router.get("/{tool_id}", response_model=Tool, operation_id="get_tool")
def get_tool(
    tool_id: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Get a tool by ID
    """
    actor = server.get_user_or_default(user_id=user_id)
    tool = server.tool_manager.get_tool_by_id(tool_id=tool_id, actor=actor)
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
    tool = server.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor)
    if tool:
        return tool.id
    else:
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
        return server.tool_manager.list_tools(actor=actor, cursor=cursor, limit=limit)
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
    try:
        actor = server.get_user_or_default(user_id=user_id)
        tool = Tool(**request.model_dump())
        return server.tool_manager.create_tool(pydantic_tool=tool, actor=actor)
    except UniqueConstraintViolationError as e:
        # Log or print the full exception here for debugging
        print(f"Error occurred: {e}")
        clean_error_message = f"Tool with name {request.name} already exists."
        raise HTTPException(status_code=409, detail=clean_error_message)
    except LettaToolCreateError as e:
        # HTTP 400 == Bad Request
        print(f"Error occurred during tool creation: {e}")
        # print the full stack trace
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch other unexpected errors and raise an internal server error
        print(f"Unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.put("/", response_model=Tool, operation_id="upsert_tool")
def upsert_tool(
    request: ToolCreate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    """
    Create or update a tool
    """
    try:
        actor = server.get_user_or_default(user_id=user_id)
        tool = server.tool_manager.create_or_update_tool(pydantic_tool=Tool(**request.model_dump()), actor=actor)
        return tool
    except UniqueConstraintViolationError as e:
        # Log the error and raise a conflict exception
        print(f"Unique constraint violation occurred: {e}")
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        # Catch other unexpected errors and raise an internal server error
        print(f"Unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


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
    actor = server.get_user_or_default(user_id=user_id)
    return server.tool_manager.update_tool_by_id(tool_id=tool_id, tool_update=request, actor=actor)


@router.post("/add-base-tools", response_model=List[Tool], operation_id="add_base_tools")
def add_base_tools(
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Add base tools
    """
    actor = server.get_user_or_default(user_id=user_id)
    return server.tool_manager.add_base_tools(actor=actor)


@router.post("/{tool_id}/run", response_model=FunctionReturn, operation_id="run_tool")
def run_tool(
    server: SyncServer = Depends(get_letta_server),
    request: ToolRun = Body(...),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Run an existing tool on provided arguments
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.run_tool(tool_id=request.tool_id, tool_args=request.tool_args, user_id=actor.id)


@router.post("/run", response_model=FunctionReturn, operation_id="run_tool_from_source")
def run_tool_from_source(
    server: SyncServer = Depends(get_letta_server),
    request: ToolRunFromSource = Body(...),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Attempt to build a tool from source, then run it on the provided arguments
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.run_tool_from_source(
        tool_source=request.source_code,
        tool_source_type=request.source_type,
        tool_args=request.args,
        tool_name=request.name,
        user_id=actor.id,
    )


# Specific routes for Composio


@router.get("/composio/apps", response_model=List[AppModel], operation_id="list_composio_apps")
def list_composio_apps(server: SyncServer = Depends(get_letta_server)):
    """
    Get a list of all Composio apps
    """
    return server.get_composio_apps()


@router.get("/composio/apps/{composio_app_name}/actions", response_model=List[ActionModel], operation_id="list_composio_actions_by_app")
def list_composio_actions_by_app(
    composio_app_name: str,
    server: SyncServer = Depends(get_letta_server),
):
    """
    Get a list of all Composio actions for a specific app
    """
    return server.get_composio_actions_from_app_name(composio_app_name=composio_app_name)


@router.post("/composio/{composio_action_name}", response_model=Tool, operation_id="add_composio_tool")
def add_composio_tool(
    composio_action_name: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    """
    Add a new Composio tool by action name (Composio refers to each tool as an `Action`)
    """
    actor = server.get_user_or_default(user_id=user_id)
    tool_create = ToolCreate.from_composio(action=composio_action_name)
    return server.tool_manager.create_or_update_tool(pydantic_tool=Tool(**tool_create.model_dump()), actor=actor)
