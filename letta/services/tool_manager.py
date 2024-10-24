from typing import List, Optional

from letta.functions.functions import derive_openai_json_schema

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
    def create_tool(self, tool_create: ToolCreate) -> PydanticTool:
        """Create a new tool based on the ToolCreate schema."""
        # Derive json_schema
        json_schema = tool_create.json_schema
        if not json_schema:
            json_schema = derive_openai_json_schema(tool_create)

        # Make tool
        with self.session_maker() as session:
            tool = ToolModel(
                user_id=tool_create.user_id,
                organization_id=tool_create.organization_id,
                description=tool_create.description,
                tags=tool_create.tags,
                source_code=tool_create.source_code,
                source_type=tool_create.source_type,
                json_schema=json_schema,
                name=json_schema["name"],
            )
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
    def tool_with_name_and_user_id_exists(self, tool_name: str, user_id: str) -> bool:
        try:
            self.get_tool_by_name_and_user_id(tool_name, user_id)
            return True
        except NoResultFound:
            return False

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
            update_data = tool_update.dict(exclude_unset=True)
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
