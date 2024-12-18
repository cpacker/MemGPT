from tqdm import tqdm

from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.tool_manager import ToolManager


def deprecated_tool():
    return "this is a deprecated tool, please remove it from your tools list"


orgs = OrganizationManager().list_organizations(cursor=None, limit=5000)
for org in tqdm(orgs):
    if org.name != "default":
        fake_user = User(id="user-00000000-0000-4000-8000-000000000000", name="fake", organization_id=org.id)

        ToolManager().upsert_base_tools(actor=fake_user)
