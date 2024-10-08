from typing import List, Literal, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from letta.schemas.tool import Tool as ToolModel  # TODO: modify
from letta.server.rest_api.interface import QueuingInterface
from letta.server.server import SyncServer

router = APIRouter()


class ListToolsResponse(BaseModel):
    tools: List[ToolModel] = Field(..., description="List of tools (functions).")


class CreateToolRequest(BaseModel):
    json_schema: dict = Field(..., description="JSON schema of the tool.")
    source_code: str = Field(..., description="The source code of the function.")
    source_type: Optional[Literal["python"]] = Field(None, description="The type of the source code.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")


class CreateToolResponse(BaseModel):
    tool: ToolModel = Field(..., description="Information about the newly created tool.")


def setup_tools_index_router(server: SyncServer, interface: QueuingInterface):

    @router.delete("/tools/{tool_name}", tags=["tools"])
    async def delete_tool(
        tool_name: str,
        # user_id: uuid.UUID = Depends(get_current_user_with_server), # TODO: add back when user-specific
    ):
        """
        Delete a tool by name
        """
        # Clear the interface
        interface.clear()
        # tool = server.ms.delete_tool(user_id=user_id, tool_name=tool_name) TODO: add back when user-specific
        server.ms.delete_tool(name=tool_name, user_id=None)

    @router.get("/tools/{tool_name}", tags=["tools"], response_model=ToolModel)
    async def get_tool(tool_name: str):
        """
        Get a tool by name
        """
        # Clear the interface
        interface.clear()
        # tool = server.ms.get_tool(user_id=user_id, tool_name=tool_name) TODO: add back when user-specific
        tool = server.ms.get_tool(tool_name=tool_name, user_id=None)
        if tool is None:
            # return 404 error
            raise HTTPException(status_code=404, detail=f"Tool with name {tool_name} not found.")
        return tool

    @router.get("/tools", tags=["tools"], response_model=ListToolsResponse)
    async def list_all_tools():
        """
        Get a list of all tools available to agents created by a user
        """
        # Clear the interface
        interface.clear()
        # tools = server.ms.list_tools(user_id=user_id) TODO: add back when user-specific
        tools = server.ms.list_tools(user_id=None)
        return ListToolsResponse(tools=tools)

    @router.post("/tools", tags=["tools"], response_model=ToolModel)
    async def create_tool(
        request: CreateToolRequest = Body(...),
    ):
        """
        Create a new tool
        """
        try:
            return server.create_tool(
                json_schema=request.json_schema, source_code=request.source_code, source_type=request.source_type, tags=request.tags
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create tool: {e}")

    return router
