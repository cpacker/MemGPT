from pytest import mark as m, fixture
from tests.mock_factory.models import (
    MockUserFactory,
    MockOrganizationFactory,
    MockTokenFactory,
)
from memgpt.orm.organization import Organization


class TestORMBases:
    """eyeball unit tests of accessors, id logic etc"""

    def test_prefixed_ids(self, db_session):

        user = MockUserFactory(db_session=db_session).generate()

        assert user.id.startswith('user-')
        assert str(user._id) in user.id
        
        with db_session as session:
            session.add(user)
            assert user.organization.id.startswith('organization-'), "Organization id is prefixed incorrectly"
            assert str(user.organization._id) in user.organization.id, "Organization id is not using the correct uuid"


@m.describe("When performing basic interactions with models")
class TestORMCRUD:
    @m.context("and reading a model")
    @m.it("should return the model")
    @fixture(
        params=[{"model": MockUserFactory},
                {"model": MockOrganizationFactory},
                {"model": MockTokenFactory}
                ],
    )
    def test_read(self, request, db_session):
        mockModel = request.param["model"]
        model = mockModel(db_session=db_session).generate()
        obj = mockModel.__model__.read(model.id, db_session)
        assert obj.id == model.id