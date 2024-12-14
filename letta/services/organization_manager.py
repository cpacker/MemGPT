from typing import List, Optional

from letta.orm.errors import NoResultFound
from letta.orm.organization import Organization as OrganizationModel
from letta.schemas.organization import Organization as PydanticOrganization
from letta.utils import enforce_types


class OrganizationManager:
    """Manager class to handle business logic related to Organizations."""

    DEFAULT_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
    DEFAULT_ORG_NAME = "default_org"

    def __init__(self):
        # TODO: Please refactor this out
        # I am currently working on a ORM refactor and would like to make a more minimal set of changes
        # - Matt
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def get_default_organization(self) -> PydanticOrganization:
        """Fetch the default organization."""
        return self.get_organization_by_id(self.DEFAULT_ORG_ID)

    @enforce_types
    def get_organization_by_id(self, org_id: str) -> Optional[PydanticOrganization]:
        """Fetch an organization by ID."""
        with self.session_maker() as session:
            organization = OrganizationModel.read(db_session=session, identifier=org_id)
            return organization.to_pydantic()

    @enforce_types
    def create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        """Create a new organization."""
        try:
            org = self.get_organization_by_id(pydantic_org.id)
            return org
        except NoResultFound:
            return self._create_organization(pydantic_org=pydantic_org)

    @enforce_types
    def _create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        with self.session_maker() as session:
            org = OrganizationModel(**pydantic_org.model_dump())
            org.create(session)
            return org.to_pydantic()

    @enforce_types
    def create_default_organization(self) -> PydanticOrganization:
        """Create the default organization."""
        return self.create_organization(PydanticOrganization(name=self.DEFAULT_ORG_NAME, id=self.DEFAULT_ORG_ID))

    @enforce_types
    def update_organization_name_using_id(self, org_id: str, name: Optional[str] = None) -> PydanticOrganization:
        """Update an organization."""
        with self.session_maker() as session:
            org = OrganizationModel.read(db_session=session, identifier=org_id)
            if name:
                org.name = name
            org.update(session)
            return org.to_pydantic()

    @enforce_types
    def delete_organization_by_id(self, org_id: str):
        """Delete an organization by marking it as deleted."""
        with self.session_maker() as session:
            organization = OrganizationModel.read(db_session=session, identifier=org_id)
            organization.hard_delete(session)

    @enforce_types
    def list_organizations(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticOrganization]:
        """List organizations with pagination based on cursor (org_id) and limit."""
        with self.session_maker() as session:
            results = OrganizationModel.list(db_session=session, cursor=cursor, limit=limit)
            return [org.to_pydantic() for org in results]
