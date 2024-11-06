from letta.functions.functions import parse_source_code
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.tool_manager import ToolManager


def deprecated_tool():
    return "this is a deprecated tool, please remove it from your tools list"


orgs = OrganizationManager().list_organizations(cursor=None, limit=5000)
for org in orgs:
    if org.name != "default":
        fake_user = User(id="user-00000000-0000-4000-8000-000000000000", name="fake", organization_id=org.id)

        ToolManager().add_base_tools(actor=fake_user)

        source_code = parse_source_code(deprecated_tool)
        source_type = "python"
        description = "deprecated"
        tags = ["deprecated"]

        ToolManager().create_or_update_tool(
            Tool(
                name="core_memory_append",
                source_code=source_code,
                source_type=source_type,
                description=description,
            ),
            actor=fake_user,
        )

        ToolManager().create_or_update_tool(
            Tool(
                name="core_memory_replace",
                source_code=source_code,
                source_type=source_type,
                description=description,
            ),
            actor=fake_user,
        )

        ToolManager().create_or_update_tool(
            Tool(
                name="pause_heartbeats",
                source_code=source_code,
                source_type=source_type,
                description=description,
            ),
            actor=fake_user,
        )
