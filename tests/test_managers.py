import pytest
from sqlalchemy import delete

import letta.utils as utils
from letta.functions.functions import derive_openai_json_schema, parse_source_code
from letta.orm.organization import Organization
from letta.orm.tool import Tool
from letta.orm.user import User
from letta.schemas.organization import Organization as PydanticOrganization
from letta.schemas.tool import Tool as PydanticTool
from letta.schemas.tool import ToolUpdate
from letta.services.organization_manager import OrganizationManager

utils.DEBUG = True
from letta.config import LettaConfig
from letta.schemas.user import User as PydanticUser
from letta.schemas.user import UserUpdate
from letta.server.server import SyncServer


@pytest.fixture(autouse=True)
def clear_tables(server: SyncServer):
    """Fixture to clear the organization table before each test."""
    with server.organization_manager.session_maker() as session:
        session.execute(delete(Block))
        session.execute(delete(Tool))  # Clear all records from the Tool table
        session.execute(delete(User))  # Clear all records from the user table
        session.execute(delete(Organization))  # Clear all records from the organization table
        session.commit()  # Commit the deletion


@pytest.fixture
def tool_fixture(server: SyncServer):
    """Fixture to create a tool with default settings and clean up after the test."""

    def print_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.
        """
        print(message)
        return message

    source_code = parse_source_code(print_tool)
    source_type = "python"
    description = "test_description"
    tags = ["test"]

    org = server.organization_manager.create_default_organization()
    user = server.user_manager.create_default_user()
    other_user = server.user_manager.create_user(PydanticUser(name="other", organization_id=org.id))
    tool = PydanticTool(description=description, tags=tags, source_code=source_code, source_type=source_type)
    derived_json_schema = derive_openai_json_schema(source_code=tool.source_code, name=tool.name)
    derived_name = derived_json_schema["name"]
    tool.json_schema = derived_json_schema
    tool.name = derived_name

    tool = server.tool_manager.create_tool(tool, actor=user)

    # Yield the created tool, organization, and user for use in tests
    yield {"tool": tool, "organization": org, "user": user, "other_user": other_user, "tool_create": tool}


@pytest.fixture(scope="module")
def server():
    config = LettaConfig.load()

    config.save()

    server = SyncServer(init_with_default_org_and_user=False)
    return server


# ======================================================================================================================
# Organization Manager Tests
# ======================================================================================================================
def test_list_organizations(server: SyncServer):
    # Create a new org and confirm that it is created correctly
    org_name = "test"
    org = server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name=org_name))

    orgs = server.organization_manager.list_organizations()
    assert len(orgs) == 1
    assert orgs[0].name == org_name

    # Delete it after
    server.organization_manager.delete_organization_by_id(org.id)
    assert len(server.organization_manager.list_organizations()) == 0


def test_create_default_organization(server: SyncServer):
    server.organization_manager.create_default_organization()
    retrieved = server.organization_manager.get_default_organization()
    assert retrieved.name == server.organization_manager.DEFAULT_ORG_NAME


def test_update_organization_name(server: SyncServer):
    org_name_a = "a"
    org_name_b = "b"
    org = server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name=org_name_a))
    assert org.name == org_name_a
    org = server.organization_manager.update_organization_name_using_id(org_id=org.id, name=org_name_b)
    assert org.name == org_name_b


def test_list_organizations_pagination(server: SyncServer):
    server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name="a"))
    server.organization_manager.create_organization(pydantic_org=PydanticOrganization(name="b"))

    orgs_x = server.organization_manager.list_organizations(limit=1)
    assert len(orgs_x) == 1

    orgs_y = server.organization_manager.list_organizations(cursor=orgs_x[0].id, limit=1)
    assert len(orgs_y) == 1
    assert orgs_y[0].name != orgs_x[0].name

    orgs = server.organization_manager.list_organizations(cursor=orgs_y[0].id, limit=1)
    assert len(orgs) == 0


# ======================================================================================================================
# User Manager Tests
# ======================================================================================================================
def test_list_users(server: SyncServer):
    # Create default organization
    org = server.organization_manager.create_default_organization()

    user_name = "user"
    user = server.user_manager.create_user(PydanticUser(name=user_name, organization_id=org.id))

    users = server.user_manager.list_users()
    assert len(users) == 1
    assert users[0].name == user_name

    # Delete it after
    server.user_manager.delete_user_by_id(user.id)
    assert len(server.user_manager.list_users()) == 0


def test_create_default_user(server: SyncServer):
    org = server.organization_manager.create_default_organization()
    server.user_manager.create_default_user(org_id=org.id)
    retrieved = server.user_manager.get_default_user()
    assert retrieved.name == server.user_manager.DEFAULT_USER_NAME


def test_update_user(server: SyncServer):
    # Create default organization
    default_org = server.organization_manager.create_default_organization()
    test_org = server.organization_manager.create_organization(PydanticOrganization(name="test_org"))

    user_name_a = "a"
    user_name_b = "b"

    # Assert it's been created
    user = server.user_manager.create_user(PydanticUser(name=user_name_a, organization_id=default_org.id))
    assert user.name == user_name_a

    # Adjust name
    user = server.user_manager.update_user(UserUpdate(id=user.id, name=user_name_b))
    assert user.name == user_name_b
    assert user.organization_id == OrganizationManager.DEFAULT_ORG_ID

    # Adjust org id
    user = server.user_manager.update_user(UserUpdate(id=user.id, organization_id=test_org.id))
    assert user.name == user_name_b
    assert user.organization_id == test_org.id


# ======================================================================================================================
# Tool Manager Tests
# ======================================================================================================================
def test_create_tool(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    tool_create = tool_fixture["tool_create"]
    user = tool_fixture["user"]
    org = tool_fixture["organization"]

    # Assertions to ensure the created tool matches the expected values
    assert tool.created_by_id == user.id
    assert tool.organization_id == org.id
    assert tool.description == tool_create.description
    assert tool.tags == tool_create.tags
    assert tool.source_code == tool_create.source_code
    assert tool.source_type == tool_create.source_type
    assert tool.json_schema == derive_openai_json_schema(source_code=tool_create.source_code, name=tool_create.name)


def test_get_tool_by_id(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]

    # Fetch the tool by ID using the manager method
    fetched_tool = server.tool_manager.get_tool_by_id(tool.id, actor=user)

    # Assertions to check if the fetched tool matches the created tool
    assert fetched_tool.id == tool.id
    assert fetched_tool.name == tool.name
    assert fetched_tool.description == tool.description
    assert fetched_tool.tags == tool.tags
    assert fetched_tool.source_code == tool.source_code
    assert fetched_tool.source_type == tool.source_type


def test_get_tool_with_actor(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]

    # Fetch the tool by name and organization ID
    fetched_tool = server.tool_manager.get_tool_by_name(tool.name, actor=user)

    # Assertions to check if the fetched tool matches the created tool
    assert fetched_tool.id == tool.id
    assert fetched_tool.name == tool.name
    assert fetched_tool.created_by_id == user.id
    assert fetched_tool.description == tool.description
    assert fetched_tool.tags == tool.tags
    assert fetched_tool.source_code == tool.source_code
    assert fetched_tool.source_type == tool.source_type


def test_list_tools(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]

    # List tools (should include the one created by the fixture)
    tools = server.tool_manager.list_tools(actor=user)

    # Assertions to check that the created tool is listed
    assert len(tools) == 1
    assert any(t.id == tool.id for t in tools)


def test_update_tool_by_id(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]
    updated_description = "updated_description"

    # Create a ToolUpdate object to modify the tool's description
    tool_update = ToolUpdate(description=updated_description)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=user)

    # Assertions to check if the update was successful
    assert updated_tool.description == updated_description


def test_update_tool_source_code_refreshes_schema_and_name(server: SyncServer, tool_fixture):
    def counter_tool(counter: int):
        """
        Args:
            counter (int): The counter to count to.

        Returns:
            bool: If it successfully counted to the counter.
        """
        for c in range(counter):
            print(c)

        return True

    # Test begins
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]
    og_json_schema = tool_fixture["tool_create"].json_schema

    source_code = parse_source_code(counter_tool)

    # Create a ToolUpdate object to modify the tool's source_code
    tool_update = ToolUpdate(source_code=source_code)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=user)

    # Assertions to check if the update was successful, and json_schema is updated as well
    assert updated_tool.source_code == source_code
    assert updated_tool.json_schema != og_json_schema

    new_schema = derive_openai_json_schema(source_code=updated_tool.source_code, name=updated_tool.name)
    assert updated_tool.json_schema == new_schema
    assert updated_tool.name == new_schema["name"]


def test_update_tool_source_code_refreshes_schema_only(server: SyncServer, tool_fixture):
    def counter_tool(counter: int):
        """
        Args:
            counter (int): The counter to count to.

        Returns:
            bool: If it successfully counted to the counter.
        """
        for c in range(counter):
            print(c)

        return True

    # Test begins
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]
    og_json_schema = tool_fixture["tool_create"].json_schema

    source_code = parse_source_code(counter_tool)
    name = "test_function_name_explicit"

    # Create a ToolUpdate object to modify the tool's source_code
    tool_update = ToolUpdate(name=name, source_code=source_code)

    # Update the tool using the manager method
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=user)

    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=user)

    # Assertions to check if the update was successful, and json_schema is updated as well
    assert updated_tool.source_code == source_code
    assert updated_tool.json_schema != og_json_schema

    new_schema = derive_openai_json_schema(source_code=updated_tool.source_code, name=updated_tool.name)
    assert updated_tool.json_schema == new_schema
    assert updated_tool.name == name


def test_update_tool_multi_user(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]
    other_user = tool_fixture["other_user"]
    updated_description = "updated_description"

    # Create a ToolUpdate object to modify the tool's description
    tool_update = ToolUpdate(description=updated_description)

    # Update the tool using the manager method, but WITH THE OTHER USER'S ID!
    server.tool_manager.update_tool_by_id(tool.id, tool_update, actor=other_user)

    # Check that the created_by and last_updated_by fields are correct
    # Fetch the updated tool to verify the changes
    updated_tool = server.tool_manager.get_tool_by_id(tool.id, actor=user)

    assert updated_tool.last_updated_by_id == other_user.id
    assert updated_tool.created_by_id == user.id


def test_delete_tool_by_id(server: SyncServer, tool_fixture):
    tool = tool_fixture["tool"]
    user = tool_fixture["user"]

    # Delete the tool using the manager method
    server.tool_manager.delete_tool_by_id(tool.id, actor=user)

    tools = server.tool_manager.list_tools(actor=user)
    assert len(tools) == 0


# ======================================================================================================================
# Block Manager Tests
# ======================================================================================================================


def test_create_block(server: SyncServer, tool_fixture):
    user = tool_fixture["user"]
    block_manager = BlockManager()
    block_create = BlockCreate(
        label="human",
        is_template=True,
        value="Sample content",
        template_name="sample_template",
        description="A test block",
        limit=1000,
        metadata_={"example": "data"},
    )

    block = block_manager.create_block(block_create, actor=user)

    # Assertions to ensure the created block matches the expected values
    assert block.label == block_create.label
    assert block.is_template == block_create.is_template
    assert block.value == block_create.value
    assert block.template_name == block_create.template_name
    assert block.description == block_create.description
    assert block.limit == block_create.limit
    assert block.metadata_ == block_create.metadata_
    assert block.organization_id == user.organization_id


def test_get_blocks(server, tool_fixture):
    user = tool_fixture["user"]
    block_manager = BlockManager()

    # Create blocks to retrieve later
    block_manager.create_block(BlockCreate(label="human", value="Block 1"), actor=user)
    block_manager.create_block(BlockCreate(label="persona", value="Block 2"), actor=user)

    # Retrieve blocks by different filters
    all_blocks = block_manager.get_blocks(actor=user)
    assert len(all_blocks) == 2

    human_blocks = block_manager.get_blocks(actor=user, label="human")
    assert len(human_blocks) == 1
    assert human_blocks[0].label == "human"

    persona_blocks = block_manager.get_blocks(actor=user, label="persona")
    assert len(persona_blocks) == 1
    assert persona_blocks[0].label == "persona"


def test_update_block(server: SyncServer, tool_fixture):
    user = tool_fixture["user"]
    block_manager = BlockManager()
    block = block_manager.create_block(BlockCreate(label="persona", value="Original Content"), actor=user)

    # Update block's content
    update_data = BlockUpdate(value="Updated Content", description="Updated description")
    block_manager.update_block(block_id=block.id, block_update=update_data, actor=user)

    # Retrieve the updated block
    updated_block = block_manager.get_blocks(actor=user, id=block.id)[0]

    # Assertions to verify the update
    assert updated_block.value == "Updated Content"
    assert updated_block.description == "Updated description"


def test_delete_block(server: SyncServer, tool_fixture):
    user = tool_fixture["user"]
    block_manager = BlockManager()

    # Create and delete a block
    block = block_manager.create_block(BlockCreate(label="human", value="Sample content"), actor=user)
    block_manager.delete_block(block_id=block.id, actor=user)

    # Verify that the block was deleted
    blocks = block_manager.get_blocks(actor=user)
    assert len(blocks) == 0
