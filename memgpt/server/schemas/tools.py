from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from memgpt.schemas.tool import Tool


class ListToolsResponse(BaseModel):
    tools: List[Tool] = Field(..., description="List of tools (functions).")


class CreateToolRequest(BaseModel):
    json_schema: dict = Field(..., description="JSON schema of the tool.")
    source_code: str = Field(..., description="The source code of the function.")
    source_type: Optional[Literal["python"]] = Field(None, description="The type of the source code.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")
    update: Optional[bool] = Field(False, description="Update the tool if it already exists.")


class CreateToolResponse(BaseModel):
    tool: Tool = Field(..., description="Information about the newly created tool.")
