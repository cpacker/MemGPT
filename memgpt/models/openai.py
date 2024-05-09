from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str = Field(..., description="The name of the function.")
    arguments: str = Field(..., description="The arguments of the function.")


class ToolCall(BaseModel):
    id: str = Field(..., description="The unique identifier of the tool call.")
    type: str = "function"
    function: Function = Field(..., description="The function call.")
