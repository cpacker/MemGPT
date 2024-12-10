from typing import Dict, List, Optional

from pydantic import Field, model_validator

from letta.constants import FUNCTION_RETURN_CHAR_LIMIT
from letta.functions.functions import derive_openai_json_schema
from letta.functions.helpers import (
    generate_composio_tool_wrapper,
    generate_langchain_tool_wrapper,
)
from letta.functions.schema_generator import generate_schema_from_args_schema_v2
from letta.schemas.letta_base import LettaBase
from letta.schemas.openai.chat_completions import ToolCall


class BaseTool(LettaBase):
    __id_prefix__ = "tool"


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
    description: Optional[str] = Field(None, description="The description of the tool.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    module: Optional[str] = Field(None, description="The module of the function.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the tool.")
    name: Optional[str] = Field(None, description="The name of the function.")
    tags: List[str] = Field([], description="Metadata tags.")

    # code
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Optional[Dict] = Field(None, description="The JSON schema of the function.")

    # tool configuration
    return_char_limit: int = Field(FUNCTION_RETURN_CHAR_LIMIT, description="The maximum number of characters in the response.")

    # metadata fields
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")

    @model_validator(mode="after")
    def populate_missing_fields(self):
        """
        Populate missing fields: name, description, and json_schema.
        """
        # Derive JSON schema if not provided
        if not self.json_schema:
            self.json_schema = derive_openai_json_schema(source_code=self.source_code)

        # Derive name from the JSON schema if not provided
        if not self.name:
            # TODO: This in theory could error, but name should always be on json_schema
            # TODO: Make JSON schema a typed pydantic object
            self.name = self.json_schema.get("name")

        # Derive description from the JSON schema if not provided
        if not self.description:
            # TODO: This in theory could error, but description should always be on json_schema
            # TODO: Make JSON schema a typed pydantic object
            self.description = self.json_schema.get("description")

        return self

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


class ToolCreate(LettaBase):
    name: Optional[str] = Field(None, description="The name of the function (auto-generated from source_code if not provided).")
    description: Optional[str] = Field(None, description="The description of the tool.")
    tags: List[str] = Field([], description="Metadata tags.")
    module: Optional[str] = Field(None, description="The source code of the function.")
    source_code: str = Field(..., description="The source code of the function.")
    source_type: str = Field("python", description="The source type of the function.")
    json_schema: Optional[Dict] = Field(
        None, description="The JSON schema of the function (auto-generated from source_code if not provided)"
    )
    return_char_limit: int = Field(FUNCTION_RETURN_CHAR_LIMIT, description="The maximum number of characters in the response.")

    @classmethod
    def from_composio(cls, action_name: str, api_key: Optional[str] = None) -> "ToolCreate":
        """
        Class method to create an instance of Letta-compatible Composio Tool.
        Check https://docs.composio.dev/introduction/intro/overview to look at options for from_composio

        This function will error if we find more than one tool, or 0 tools.

        Args:
            action_name str: A action name to filter tools by.
        Returns:
            Tool: A Letta Tool initialized with attributes derived from the Composio tool.
        """
        from composio import LogLevel
        from composio_langchain import ComposioToolSet

        if api_key:
            # Pass in an external API key
            composio_toolset = ComposioToolSet(logging_level=LogLevel.ERROR, api_key=api_key)
        else:
            # Use environmental variable
            composio_toolset = ComposioToolSet(logging_level=LogLevel.ERROR)
        composio_tools = composio_toolset.get_tools(actions=[action_name])

        assert len(composio_tools) > 0, "User supplied parameters do not match any Composio tools"
        assert len(composio_tools) == 1, f"User supplied parameters match too many Composio tools; {len(composio_tools)} > 1"

        composio_tool = composio_tools[0]

        description = composio_tool.description
        source_type = "python"
        tags = ["composio"]
        wrapper_func_name, wrapper_function_str = generate_composio_tool_wrapper(action_name)
        json_schema = generate_schema_from_args_schema_v2(composio_tool.args_schema, name=wrapper_func_name, description=description)

        return cls(
            name=wrapper_func_name,
            description=description,
            source_type=source_type,
            tags=tags,
            source_code=wrapper_function_str,
            json_schema=json_schema,
        )

    @classmethod
    def from_langchain(
        cls,
        langchain_tool: "LangChainBaseTool",
        additional_imports_module_attr_map: dict[str, str] = None,
    ) -> "ToolCreate":
        """
        Class method to create an instance of Tool from a Langchain tool (must be from langchain_community.tools).

        Args:
            langchain_tool (LangChainBaseTool): An instance of a LangChain BaseTool (BaseTool from LangChain)
            additional_imports_module_attr_map (dict[str, str]): A mapping of module names to attribute name. This is used internally to import all the required classes for the langchain tool. For example, you would pass in `{"langchain_community.utilities": "WikipediaAPIWrapper"}` for `from langchain_community.tools import WikipediaQueryRun`. NOTE: You do NOT need to specify the tool import here, that is done automatically for you.

        Returns:
            Tool: A Letta Tool initialized with attributes derived from the provided LangChain BaseTool object.
        """
        description = langchain_tool.description
        source_type = "python"
        tags = ["langchain"]
        # NOTE: langchain tools may come from different packages
        wrapper_func_name, wrapper_function_str = generate_langchain_tool_wrapper(langchain_tool, additional_imports_module_attr_map)
        json_schema = generate_schema_from_args_schema_v2(langchain_tool.args_schema, name=wrapper_func_name, description=description)

        return cls(
            name=wrapper_func_name,
            description=description,
            source_type=source_type,
            tags=tags,
            source_code=wrapper_function_str,
            json_schema=json_schema,
        )

    @classmethod
    def load_default_langchain_tools(cls) -> List["ToolCreate"]:
        # For now, we only support wikipedia tool
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wikipedia_tool = ToolCreate.from_langchain(
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()), {"langchain_community.utilities": "WikipediaAPIWrapper"}
        )

        return [wikipedia_tool]

    @classmethod
    def load_default_composio_tools(cls) -> List["ToolCreate"]:
        pass

        # TODO: Disable composio tools for now
        # TODO: Naming is causing issues
        # calculator = ToolCreate.from_composio(action_name=Action.MATHEMATICAL_CALCULATOR.name)
        # serp_news = ToolCreate.from_composio(action_name=Action.SERPAPI_NEWS_SEARCH.name)
        # serp_google_search = ToolCreate.from_composio(action_name=Action.SERPAPI_SEARCH.name)
        # serp_google_maps = ToolCreate.from_composio(action_name=Action.SERPAPI_GOOGLE_MAPS_SEARCH.name)

        return []


class ToolUpdate(LettaBase):
    description: Optional[str] = Field(None, description="The description of the tool.")
    name: Optional[str] = Field(None, description="The name of the function.")
    tags: Optional[List[str]] = Field(None, description="Metadata tags.")
    module: Optional[str] = Field(None, description="The source code of the function.")
    source_code: Optional[str] = Field(None, description="The source code of the function.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    json_schema: Optional[Dict] = Field(
        None, description="The JSON schema of the function (auto-generated from source_code if not provided)"
    )

    class Config:
        extra = "ignore"  # Allows extra fields without validation errors
        # TODO: Remove this, and clean usage of ToolUpdate everywhere else


class ToolRun(LettaBase):
    id: str = Field(..., description="The ID of the tool to run.")
    args: str = Field(..., description="The arguments to pass to the tool (as stringified JSON).")


class ToolRunFromSource(LettaBase):
    source_code: str = Field(..., description="The source code of the function.")
    args: str = Field(..., description="The arguments to pass to the tool (as stringified JSON).")
    name: Optional[str] = Field(None, description="The name of the tool to run.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
