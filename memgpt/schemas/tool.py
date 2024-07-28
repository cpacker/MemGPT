from typing import Dict, List, Optional

from pydantic import Field

from memgpt.schemas.memgpt_base import MemGPTBase
from memgpt.schemas.openai.chat_completions import ToolCall


class BaseTool(MemGPTBase):
    __id_prefix__ = "tool"

    # optional fields
    description: Optional[str] = Field(None, description="The description of the tool.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    module: Optional[str] = Field(None, description="The module of the function.")

    # optional: user_id (user-specific tools)
    user_id: Optional[str] = Field(None, description="The unique identifier of the user associated with the function.")


class Tool(BaseTool):

    id: str = BaseTool.generate_id_field()

    name: str = Field(..., description="The name of the function.")
    tags: List[str] = Field(..., description="Metadata tags.")

    # code
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Dict = Field(default_factory=dict, description="The JSON schema of the function.")

    def to_dict(self):
        """Convert into OpenAI representation"""
        return vars(
            ToolCall(
                tool_id=self.id,
                tool_call_type="function",
                function=self.module,
            )
        )


class ToolCreate(BaseTool):
    name: str = Field(..., description="The name of the function.")
    tags: List[str] = Field(..., description="Metadata tags.")
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Dict = Field(default_factory=dict, description="The JSON schema of the function.")


class ToolUpdate(ToolCreate):
    id: str = Field(..., description="The unique identifier of the tool.")
    name: Optional[str] = Field(None, description="The name of the function.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")
    source_code: Optional[str] = Field(None, description="The source code of the function.")
    json_schema: Optional[Dict] = Field(None, description="The JSON schema of the function.")
