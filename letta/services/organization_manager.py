from typing import List, Optional

from sqlalchemy.exc import NoResultFound

from letta.orm.organization import Organization
from letta.schemas.organization import Organization as PydanticOrganization
from letta.utils import create_random_username


class OrganizationManager:
    """Manager class to handle business logic related to Organizations."""

    def __init__(self):
        # This is probably horrible but we reuse this technique from metadata.py
        # TODO: Please refactor this out
        # I am currently working on a ORM refactor and would like to make a more minimal set of changes
        # - Matt
        from letta.server.server import db_context

        self.session_maker = db_context

    def get_organization_by_id(self, org_id: str) -> PydanticOrganization:
        """Fetch an organization by ID."""
        with self.session_maker() as session:
            try:
                organization = Organization.read(db_session=session, identifier=org_id)
                return organization.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Organization with id {org_id} not found.")

    def create_organization(self, name: Optional[str] = None, org_id: Optional[str] = None) -> PydanticOrganization:
        """Create a new organization. If org_id is provided, it uses it, otherwise generates a new one."""
        if not name:
            name = create_random_username()

        with self.session_maker() as session:
            # Create an organization, setting the ID if provided, otherwise generating a new one
            org = Organization(name=name)

            if org_id:
                org.id = org_id  # This will trigger the setter logic for validating and assigning the id

            org.create(session)

            return org.to_pydantic()

    def update_organization(self, org_id: str, name: Optional[str] = None) -> PydanticOrganization:
        """Update an organization."""
        with self.session_maker() as session:
            organization = Organization.read(db_session=session, identifier=org_id)
            if name:
                organization.name = name
            organization.update(session)
            return organization.to_pydantic()

    def delete_organization(self, org_id: str):
        """Delete an organization by marking it as deleted."""
        with self.session_maker() as session:
            organization = Organization.read(db_session=session, identifier=org_id)
            organization.delete(session)

    def list_organizations(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticOrganization]:
        """List organizations with pagination based on cursor (org_id) and limit."""
        with self.session_maker() as session:
            # query = select(Organization)
            #
            # # If a cursor (org_id) is provided, fetch organizations with IDs greater than the cursor
            # if cursor:
            #     query = query.where(Organization._id > Organization.to_uid(cursor))
            #
            # query = query.order_by(Organization._id).limit(limit)
            #
            # # Execute the query
            # results = session.execute(query).scalars().all()
            #
            results = Organization.list(db_session=session)
            return [org.to_pydantic() for org in results]
