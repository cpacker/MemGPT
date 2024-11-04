from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.tool_manager import ToolManager

orgs = OrganizationManager().list_organizations(cursor=None, limit=5000)
for org in orgs:
    if org.name != "default":
        fake_user = User(id="user-00000000-0000-4000-8000-000000000000", name="fake", organization_id=org.id)

        ToolManager().add_base_tools(actor=fake_user)
