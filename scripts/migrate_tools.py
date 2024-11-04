from letta.functions.functions import derive_openai_json_schema, parse_source_code
from letta.schemas.tool import ToolCreate
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

        tool_create = ToolCreate(
            name="core_memory_append",
            source_code=source_code,
            source_type=source_type,
            description=description,
        )

        derived_json_schema = derive_openai_json_schema(source_code=tool_create.source_code, name=tool_create.name)
        derived_name = derived_json_schema["name"]
        tool_create.json_schema = derived_json_schema

        ToolManager().create_or_update_tool(
            tool_create,
            actor=fake_user,
        )

        tool_create.name = "core_memory_replace"

        ToolManager().create_or_update_tool(
            tool_create,
            actor=fake_user,
        )
