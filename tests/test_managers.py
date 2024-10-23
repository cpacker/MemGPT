import pytest
from sqlalchemy import delete

import letta.utils as utils
from letta.constants import (
    DEFAULT_ORG_ID,
    DEFAULT_ORG_NAME,
    DEFAULT_USER_ID,
    DEFAULT_USER_NAME,
)
from letta.orm.organization import Organization
from letta.orm.user import User

utils.DEBUG = True
from letta.config import LettaConfig
from letta.schemas.user import UserCreate, UserUpdate
from letta.server.server import SyncServer


@pytest.fixture(autouse=True)
def clear_organization_and_user_table(server: SyncServer):
    """Fixture to clear the organization table before each test."""
    with server.organization_manager.session_maker() as session:
        session.execute(delete(User))  # Clear all records from the user table
        session.execute(delete(Organization))  # Clear all records from the organization table
        session.commit()  # Commit the deletion


@pytest.fixture(scope="module")
def server():
    config = LettaConfig.load()

    config.save()

    server = SyncServer()
    return server


# ======================================================================================================================
# Organization Manager Tests
# ======================================================================================================================
def test_list_organizations(server: SyncServer):
    # Create a new org and confirm that it is created correctly
    org_name = "test"
    org = server.organization_manager.create_organization(name=org_name)

    orgs = server.organization_manager.list_organizations()
    assert len(orgs) == 1
    assert orgs[0].name == org_name

    # Delete it after
    server.organization_manager.delete_organization_by_id(org.id)
    assert len(server.organization_manager.list_organizations()) == 0


def test_create_default_organization(server: SyncServer):
    server.organization_manager.create_default_organization()
    retrieved = server.organization_manager.get_organization_by_id(DEFAULT_ORG_ID)
    assert retrieved.name == DEFAULT_ORG_NAME


def test_update_organization_name(server: SyncServer):
    org_name_a = "a"
    org_name_b = "b"
    org = server.organization_manager.create_organization(name=org_name_a)
    assert org.name == org_name_a
    org = server.organization_manager.update_organization_name_using_id(org_id=org.id, name=org_name_b)
    assert org.name == org_name_b


def test_list_organizations_pagination(server: SyncServer):
    server.organization_manager.create_organization(name="a")
    server.organization_manager.create_organization(name="b")

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
    user = server.user_manager.create_user(UserCreate(name=user_name, organization_id=org.id))

    users = server.user_manager.list_users()
    assert len(users) == 1
    assert users[0].name == user_name

    # Delete it after
    server.user_manager.delete_user_by_id(user.id)
    assert len(server.user_manager.list_users()) == 0


def test_create_default_user(server: SyncServer):
    org = server.organization_manager.create_default_organization()
    server.user_manager.create_default_user(org_id=org.id)
    retrieved = server.user_manager.get_user_by_id(DEFAULT_USER_ID)
    assert retrieved.name == DEFAULT_USER_NAME


def test_update_user(server: SyncServer):
    # Create default organization
    default_org = server.organization_manager.create_default_organization()
    test_org = server.organization_manager.create_organization(name="test_org")

    user_name_a = "a"
    user_name_b = "b"

    # Assert it's been created
    user = server.user_manager.create_user(UserCreate(name=user_name_a, organization_id=default_org.id))
    assert user.name == user_name_a

    # Adjust name
    user = server.user_manager.update_user(UserUpdate(id=user.id, name=user_name_b))
    assert user.name == user_name_b
    assert user.organization_id == DEFAULT_ORG_ID

    # Adjust org id
    user = server.user_manager.update_user(UserUpdate(id=user.id, organization_id=test_org.id))
    assert user.name == user_name_b
    assert user.organization_id == test_org.id
