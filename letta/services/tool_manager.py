import importlib
import inspect
import warnings
from typing import List, Optional

from letta.functions.functions import derive_openai_json_schema, load_function_set

# TODO: Remove this once we translate all of these to the ORM
from letta.orm.errors import NoResultFound
from letta.orm.organization import Organization as OrganizationModel
from letta.orm.tool import Tool as ToolModel
from letta.orm.user import User as UserModel
from letta.schemas.tool import Tool as PydanticTool
from letta.schemas.tool import ToolCreate, ToolUpdate
from letta.utils import enforce_types


class ToolManager:
    """Manager class to handle business logic related to Tools."""

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_or_update_tool(self, tool_create: ToolCreate) -> PydanticTool:
        """Create a new tool based on the ToolCreate schema."""
        # Derive json_schema
        derived_json_schema = tool_create.json_schema or derive_openai_json_schema(tool_create)
        derived_name = tool_create.name or derived_json_schema["name"]

        try:
            # NOTE: We use the organization id here
            # This is important, because even if it's a different user, adding the same tool to the org should not happen
            tool = self.get_tool_by_name_and_org_id(tool_name=derived_name, organization_id=tool_create.organization_id)
            # Put to dict and remove fields that should not be reset
            update_data = tool_create.model_dump(exclude={"user_id", "organization_id", "module", "terminal"}, exclude_unset=True)
            # Remove redundant update fields
            update_data = {key: value for key, value in update_data.items() if getattr(tool, key) != value}

            # If there's anything to update
            if update_data:
                self.update_tool_by_id(tool.id, ToolUpdate(**update_data))
            else:
                warnings.warn(
                    f"`create_or_update_tool` was called with user_id={tool_create.user_id}, organization_id={tool_create.organization_id}, name={tool_create.name}, but found existing tool with nothing to update."
                )
        except NoResultFound:
            tool_create.json_schema = derived_json_schema
            tool_create.name = derived_name
            tool = self.create_tool(tool_create)

        return tool

    @enforce_types
    def create_tool(self, tool_create: ToolCreate) -> PydanticTool:
        """Create a new tool based on the ToolCreate schema."""
        # Create the tool
        with self.session_maker() as session:
            # Include all fields except 'terminal' (which is not part of ToolModel) at the moment
            create_data = tool_create.model_dump(exclude={"terminal"})
            tool = ToolModel(**create_data)  # Unpack everything directly into ToolModel
            tool.create(session)

        return tool.to_pydantic()

    @enforce_types
    def get_tool_by_id(self, tool_id: str) -> PydanticTool:
        """Fetch a tool by its ID."""
        with self.session_maker() as session:
            try:
                # Retrieve tool by id using the Tool model's read method
                tool = ToolModel.read(db_session=session, identifier=tool_id)
                # Convert the SQLAlchemy Tool object to PydanticTool
                return tool.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Tool with id {tool_id} not found.")

    @enforce_types
    def get_tool_by_name_and_user_id(self, tool_name: str, user_id: str) -> PydanticTool:
        """Retrieve a tool by its name and organization_id."""
        with self.session_maker() as session:
            # Use the list method to apply filters
            results = ToolModel.list(db_session=session, name=tool_name, _user_id=UserModel.get_uid_from_identifier(user_id))

            # Ensure only one result is returned (since there is a unique constraint)
            if not results:
                raise NoResultFound(f"Tool with name {tool_name} and user_id {user_id} not found.")

            if len(results) > 1:
                raise RuntimeError(
                    f"Multiple tools with name {tool_name} and user_id {user_id} were found. This is a serious error, and means that our table does not have uniqueness constraints properly set up. Please reach out to the letta development team if you see this error."
                )

            # Return the single result
            return results[0]

    @enforce_types
    def get_tool_by_name_and_org_id(self, tool_name: str, organization_id: str) -> PydanticTool:
        """Retrieve a tool by its name and organization_id."""
        with self.session_maker() as session:
            # Use the list method to apply filters
            results = ToolModel.list(
                db_session=session, name=tool_name, _organization_id=OrganizationModel.get_uid_from_identifier(organization_id)
            )

            # Ensure only one result is returned (since there is a unique constraint)
            if not results:
                raise NoResultFound(f"Tool with name {tool_name} and organization_id {organization_id} not found.")

            if len(results) > 1:
                raise RuntimeError(
                    f"Multiple tools with name {tool_name} and organization_id {organization_id} were found. This is a serious error, and means that our table does not have uniqueness constraints properly set up. Please reach out to the letta development team if you see this error."
                )

            # Return the single result
            return results[0]

    @enforce_types
    def list_tools_for_org(self, organization_id: str, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticTool]:
        """List all tools with optional pagination using cursor and limit."""
        with self.session_maker() as session:
            tools = ToolModel.list(
                db_session=session, cursor=cursor, limit=limit, _organization_id=OrganizationModel.get_uid_from_identifier(organization_id)
            )
            return [tool.to_pydantic() for tool in tools]

    @enforce_types
    def update_tool_by_id(self, tool_id: str, tool_update: ToolUpdate) -> None:
        """Update a tool by its ID with the given ToolUpdate object."""
        with self.session_maker() as session:
            # Fetch the tool by ID
            tool = ToolModel.read(db_session=session, identifier=tool_id)

            # Update tool attributes with only the fields that were explicitly set
            update_data = tool_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(tool, key, value)

            # Save the updated tool to the database
            tool.update(db_session=session)

    @enforce_types
    def delete_tool_by_id(self, tool_id: str) -> None:
        """Delete a tool by its ID."""
        with self.session_maker() as session:
            try:
                tool = ToolModel.read(db_session=session, identifier=tool_id)
                tool.delete(db_session=session)
            except NoResultFound:
                raise ValueError(f"Tool with id {tool_id} not found.")

    @enforce_types
    def add_default_tools(self, user_id: str, org_id: str, module_name="base"):
        """Add default tools in {module_name}.py"""
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
        for name, schema in functions_to_schema.items():
            # print([str(inspect.getsource(line)) for line in schema["imports"]])
            source_code = inspect.getsource(schema["python_function"])
            tags = [module_name]
            if module_name == "base":
                tags.append("letta-base")

            # create to tool
            self.create_or_update_tool(
                ToolCreate(
                    name=name,
                    tags=tags,
                    source_type="python",
                    module=schema["module"],
                    source_code=source_code,
                    json_schema=schema["json_schema"],
                    organization_id=org_id,
                    user_id=user_id,
                ),
            )
