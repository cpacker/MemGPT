from typing import List, Literal, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import ToolModel
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


def setup_tools_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    # get_current_user_with_server = partial(partial(get_current_user, server), password)

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
        server.ms.delete_tool(name=tool_name)

    @router.get("/tools/{tool_name}", tags=["tools"], response_model=ToolModel)
    async def get_tool(tool_name: str):
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
        # user_id: uuid.UUID = Depends(get_current_user_with_server), # TODO: add back when user-specific
    ):
        """
        Get a list of all tools available to agents created by a user
        """
        # Clear the interface
        interface.clear()
        # tools = server.ms.list_tools(user_id=user_id) TODO: add back when user-specific
        tools = server.ms.list_tools()
        return ListToolsResponse(tools=tools)

    @router.post("/tools", tags=["tools"], response_model=ToolModel)
    async def create_tool(
        request: CreateToolRequest = Body(...),
        # user_id: uuid.UUID = Depends(get_current_user_with_server), # TODO: add back when user-specific
    ):
        """
        Create a new tool (dummy route)
        """
        from memgpt.functions.functions import load_function_file, write_function

        # check if function already exists
        if server.ms.get_tool(request.name):
            raise ValueError(f"Tool with name {request.name} already exists.")

        # write function to ~/.memgt/functions directory
        file_path = write_function(request.name, request.name, request.source_code)

        # TODO: Use load_function_file to load function schema
        schema = load_function_file(file_path)
        assert len(list(schema.keys())) == 1, "Function schema must have exactly one key"
        json_schema = list(schema.values())[0]["json_schema"]

        print("adding tool", request.name, request.tags, request.source_code)
        tool = ToolModel(name=request.name, json_schema=json_schema, tags=request.tags, source_code=request.source_code)
        tool.id
        server.ms.add_tool(tool)

        # TODO: insert tool information into DB as ToolModel
        return server.ms.get_tool(request.name)

    return router
