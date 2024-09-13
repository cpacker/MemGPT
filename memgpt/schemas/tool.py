from typing import Dict, List, Optional

from pydantic import Field

from memgpt.functions.schema_generator import (
    generate_crewai_tool_wrapper,
    generate_langchain_tool_wrapper,
    generate_schema_from_args_schema,
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
    """
    Representation of a tool, which is a function that can be called by the agent.

    Parameters:
        id (str): The unique identifier of the tool.
        name (str): The name of the function.
        tags (List[str]): Metadata tags.
        source_code (str): The source code of the function.
        json_schema (Dict): The JSON schema of the function.

    """

    id: str = BaseTool.generate_id_field()

    name: str = Field(..., description="The name of the function.")
    tags: List[str] = Field(..., description="Metadata tags.")

    # code
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Dict = Field(default_factory=dict, description="The JSON schema of the function.")

    def to_dict(self):
        """
        Convert tool into OpenAI representation.
        """
        return vars(
            ToolCall(
                tool_id=self.id,
                tool_call_type="function",
                function=self.module,
            )
        )

    @classmethod
    def from_langchain(cls, langchain_tool) -> "Tool":
        """
        Class method to create an instance of Tool from a Langchain tool (must be from langchain_community.tools).

        Args:
            langchain_tool (LangchainTool): An instance of a crewAI BaseTool (BaseTool from crewai)

        Returns:
            Tool: A MemGPT Tool initialized with attributes derived from the provided crewAI BaseTool object.
        """
        description = langchain_tool.description
        source_type = "python"
        tags = ["langchain"]
        # NOTE: langchain tools may come from different packages
        wrapper_func_name, wrapper_function_str = generate_langchain_tool_wrapper(langchain_tool.__class__.__name__)
        json_schema = generate_schema_from_args_schema(langchain_tool.args_schema, name=wrapper_func_name, description=description)

        # append heartbeat (necessary for triggering another reasoning step after this tool call)
        json_schema["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to 'true' if you want to send a follow-up message or run a follow-up function.",
        }
        json_schema["parameters"]["required"].append("request_heartbeat")

        return cls(
            name=wrapper_func_name,
            description=description,
            source_type=source_type,
            tags=tags,
            source_code=wrapper_function_str,
            json_schema=json_schema,
        )

    @classmethod
    def from_crewai(cls, crewai_tool) -> "Tool":
        """
        Class method to create an instance of Tool from a crewAI BaseTool object.

        Args:
            crewai_tool (CrewAIBaseTool): An instance of a crewAI BaseTool (BaseTool from crewai)

        Returns:
            Tool: A MemGPT Tool initialized with attributes derived from the provided crewAI BaseTool object.
        """
        crewai_tool.name
        description = crewai_tool.description
        source_type = "python"
        tags = ["crew-ai"]
        wrapper_func_name, wrapper_function_str = generate_crewai_tool_wrapper(crewai_tool.__class__.__name__)
        json_schema = generate_schema_from_args_schema(crewai_tool.args_schema, name=wrapper_func_name, description=description)

        # append heartbeat (necessary for triggering another reasoning step after this tool call)
        json_schema["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to 'true' if you want to send a follow-up message or run a follow-up function.",
        }
        json_schema["parameters"]["required"].append("request_heartbeat")

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
