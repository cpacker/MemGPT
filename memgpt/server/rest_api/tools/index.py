import uuid
from functools import partial
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import ToolModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class ListToolsResponse(BaseModel):
    tools: List[ToolModel] = Field(..., description="List of tools (functions).")


class CreateToolRequest(BaseModel):
    name: str = Field(..., description="The name of the function.")
    source_code: str = Field(..., description="The source code of the function.")
    source_type: Optional[Literal["python"]] = Field(None, description="The type of the source code.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")


class CreateToolResponse(BaseModel):
    tool: ToolModel = Field(..., description="Information about the newly created tool.")


def setup_user_tools_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/tools/{tool_name}", tags=["tools"], response_model=ToolModel)
    async def get_tool(
        tool_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),  # TODO: add back when user-specific
    ):
        """
        Get a tool by name
        """
        # Clear the interface
        interface.clear()
        # tool = server.ms.get_tool(user_id=user_id, tool_name=tool_name) TODO: add back when user-specific
        tool = server.ms.get_tool(tool_name=tool_name)
        if tool is None:
            # return 404 error
            raise HTTPException(status_code=404, detail=f"Tool with name {tool_name} not found.")
        return tool

    @router.get("/tools", tags=["tools"], response_model=ListToolsResponse)
    async def list_all_tools(
        user_id: uuid.UUID = Depends(get_current_user_with_server),  # TODO: add back when user-specific
    ):
        """
        Get a list of all tools available to agents created by a user
        """
        # Clear the interface
        interface.clear()
        # tools = server.ms.list_tools(user_id=user_id) TODO: add back when user-specific
        tools = server.ms.list_tools()
        return ListToolsResponse(tools=tools)

    return router
