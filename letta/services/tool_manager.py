import importlib
import inspect
import warnings
from typing import List, Optional

from letta.functions.functions import derive_openai_json_schema, load_function_set

# TODO: Remove this once we translate all of these to the ORM
from letta.orm.errors import NoResultFound
from letta.orm.organization import Organization as OrganizationModel
from letta.orm.tool import Tool as ToolModel
from letta.schemas.tool import Tool as PydanticTool
from letta.schemas.tool import ToolCreate, ToolUpdate
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types


class ToolManager:
    """Manager class to handle business logic related to Tools."""

    BASE_TOOL_NAMES = [
        "send_message",
        "conversation_search",
        "conversation_search_date",
        "archival_memory_insert",
        "archival_memory_search",
    ]

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_or_update_tool(self, tool_create: ToolCreate, actor: PydanticUser) -> PydanticTool:
        """Create a new tool based on the ToolCreate schema."""
        # Derive json_schema
        derived_json_schema = tool_create.json_schema or derive_openai_json_schema(
            source_code=tool_create.source_code, name=tool_create.name
        )
        derived_name = tool_create.name or derived_json_schema["name"]

        try:
            # NOTE: We use the organization id here
            # This is important, because even if it's a different user, adding the same tool to the org should not happen
            tool = self.get_tool_by_name(tool_name=derived_name, actor=actor)
            # Put to dict and remove fields that should not be reset
            update_data = tool_create.model_dump(exclude={"module"}, exclude_unset=True)
            # Remove redundant update fields
            update_data = {key: value for key, value in update_data.items() if getattr(tool, key) != value}

            # If there's anything to update
            if update_data:
                self.update_tool_by_id(tool.id, ToolUpdate(**update_data), actor)
            else:
                warnings.warn(
                    f"`create_or_update_tool` was called with user_id={actor.id}, organization_id={actor.organization_id}, name={tool_create.name}, but found existing tool with nothing to update."
                )
        except NoResultFound:
            tool_create.json_schema = derived_json_schema
            tool_create.name = derived_name
            tool = self.create_tool(tool_create, actor=actor)

        return tool

    @enforce_types
    def create_tool(self, tool_create: ToolCreate, actor: PydanticUser) -> PydanticTool:
        """Create a new tool based on the ToolCreate schema."""
        # Create the tool
        with self.session_maker() as session:
            create_data = tool_create.model_dump()
            tool = ToolModel(**create_data, organization_id=actor.organization_id)  # Unpack everything directly into ToolModel
            tool.create(session, actor=actor)

        return tool.to_pydantic()

    @enforce_types
    def get_tool_by_id(self, tool_id: str, actor: PydanticUser) -> PydanticTool:
        """Fetch a tool by its ID."""
        with self.session_maker() as session:
            # Retrieve tool by id using the Tool model's read method
            tool = ToolModel.read(db_session=session, identifier=tool_id, actor=actor)
            # Convert the SQLAlchemy Tool object to PydanticTool
            return tool.to_pydantic()

    @enforce_types
    def get_tool_by_name(self, tool_name: str, actor: PydanticUser):
        """Retrieve a tool by its name and a user. We derive the organization from the user, and retrieve that tool."""
        with self.session_maker() as session:
            tool = ToolModel.read(db_session=session, name=tool_name, actor=actor)
            return tool.to_pydantic()

    @enforce_types
    def list_tools(self, actor: PydanticUser, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticTool]:
        """List all tools with optional pagination using cursor and limit."""
        with self.session_maker() as session:
            tools = ToolModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                _organization_id=OrganizationModel.get_uid_from_identifier(actor.organization_id),
            )
            return [tool.to_pydantic() for tool in tools]

    @enforce_types
    def update_tool_by_id(self, tool_id: str, tool_update: ToolUpdate, actor: PydanticUser) -> None:
        """Update a tool by its ID with the given ToolUpdate object."""
        with self.session_maker() as session:
            # Fetch the tool by ID
            tool = ToolModel.read(db_session=session, identifier=tool_id, actor=actor)

            # Update tool attributes with only the fields that were explicitly set
            update_data = tool_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(tool, key, value)

            # If source code is changed and a new json_schema is not provided, we want to auto-refresh the name and schema
            # CAUTION: This will override any name/schema values the user passed in
            if "source_code" in update_data.keys() and "json_schema" not in update_data.keys():
                pydantic_tool = tool.to_pydantic()

                # Decide whether or not to reset name
                # If name was not explicitly passed in as part of the update, then we auto-generate a new name based on source code
                name = None
                if "name" in update_data.keys():
                    name = update_data["name"]
                new_schema = derive_openai_json_schema(source_code=pydantic_tool.source_code, name=name)

                # The name will either be set (if explicit) or autogenerated from the source code
                tool.name = new_schema["name"]
                tool.json_schema = new_schema

            # Save the updated tool to the database
            tool.update(db_session=session, actor=actor)

    @enforce_types
    def delete_tool_by_id(self, tool_id: str, actor: PydanticUser) -> None:
        """Delete a tool by its ID."""
        with self.session_maker() as session:
            try:
                tool = ToolModel.read(db_session=session, identifier=tool_id)
                tool.delete(db_session=session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Tool with id {tool_id} not found.")

    @enforce_types
    def add_base_tools(self, actor: PydanticUser) -> List[PydanticTool]:
        """Add default tools in base.py"""
        module_name = "base"
        full_module_name = f"letta.functions.function_sets.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            # Handle other general exceptions
            raise e

        functions_to_schema = []
        try:
            # Load the function set
            functions_to_schema = load_function_set(module)
        except ValueError as e:
            err = f"Error loading function set '{module_name}': {e}"
            warnings.warn(err)

        # create tool in db
        tools = []
        for name, schema in functions_to_schema.items():
            if name in self.BASE_TOOL_NAMES:
                # print([str(inspect.getsource(line)) for line in schema["imports"]])
                source_code = inspect.getsource(schema["python_function"])
                tags = [module_name]
                if module_name == "base":
                    tags.append("letta-base")

                # create to tool
                tools.append(
                    self.create_or_update_tool(
                        ToolCreate(
                            name=name,
                            tags=tags,
                            source_type="python",
                            module=schema["module"],
                            source_code=source_code,
                            json_schema=schema["json_schema"],
                        ),
                        actor=actor,
                    )
                )

        return tools
