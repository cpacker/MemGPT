from pytest import mark as m
from faker import Faker

from memgpt.orm.user import User
from memgpt.orm.organization import Organization

faker = Faker()

@m.unit
class TestORMBases:
    """eyeball unit tests of accessors, id logic etc"""

    def test_prefixed_ids(db_session):

        user = User(
            email=faker.email,
            organization=Organization.default()
        ).create(db_session)

        assert user.id.startswith('user-')
        assert str(user._id) in user.id
        assert user.organization.id.startswith('organization-')
        assert str(user.organization._id) in user.organization.id