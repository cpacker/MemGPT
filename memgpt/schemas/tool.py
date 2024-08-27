from typing import Dict, List, Optional

from pydantic import Field

from memgpt.functions.schema_generator import (
    generate_schema_from_args_schema,
    generate_tool_wrapper,
)
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

    @classmethod
    def from_crewai(cls, crewai_tool) -> "Tool":
        """
        Class method to create an instance of Tool from a crewAI BaseTool object.

        Args:
            crewai_tool (CrewAIBaseTool): An instance of a crewAI BaseTool (BaseTool from crewai)

        Returns:
            Tool: A memGPT Tool initialized with attributes derived from the provided crewAI BaseTool object.
        """
        crewai_tool.name
        description = crewai_tool.description
        source_type = "python"
        tags = ["crew-ai"]
        wrapper_func_name, wrapper_function_str = generate_tool_wrapper(crewai_tool.__class__.__name__)
        json_schema = generate_schema_from_args_schema(crewai_tool.args_schema, name=wrapper_func_name, description=description)

        return cls(
            name=wrapper_func_name,
            description=description,
            source_type=source_type,
            tags=tags,
            source_code=wrapper_function_str,
            json_schema=json_schema,
        )


class ToolCreate(BaseTool):
    name: Optional[str] = Field(None, description="The name of the function (auto-generated from source_code if not provided).")
    tags: List[str] = Field(..., description="Metadata tags.")
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Optional[Dict] = Field(
        None, description="The JSON schema of the function (auto-generated from source_code if not provided)"
    )


class ToolUpdate(ToolCreate):
    id: str = Field(..., description="The unique identifier of the tool.")
    name: Optional[str] = Field(None, description="The name of the function.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")
    source_code: Optional[str] = Field(None, description="The source code of the function.")
    json_schema: Optional[Dict] = Field(None, description="The JSON schema of the function.")
