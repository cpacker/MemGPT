from pytest import mark as m
from tests.mock_factory.models import (
    MockUserFactory,
    MockOrganizationFactory,
    MockTokenFactory,
    MockAgentFactory,
    MockToolFactory,
)


@m.describe("When performing basic interactions with models")
class TestORM:
    @m.context("and reading a model")
    @m.it("should return the model")
    @m.parametrize(
        "mockModel",
        [
            MockUserFactory,
            MockOrganizationFactory,
            MockTokenFactory,
            MockAgentFactory,
            MockToolFactory,
        ]
    )
    def test_read_models(self, mockModel, db_session):
        model = mockModel(db_session=db_session).generate()
        match mockModel.__model__.__name__:
            case "Tool":
                obj = mockModel.__model__.read(db_session=db_session, name=model.name)
            case _:
                obj = mockModel.__model__.read(db_session=db_session, identifier=str(model._id))

        assert obj.id == model.id


    @m.context("and creating a User model")
    @m.it("should have the correct prefixes")
    def test_prefixed_ids(self, db_session):

        user = MockUserFactory(db_session=db_session).generate()

        assert user.id.startswith('user-')
        assert str(user._id) in user.id
        
        with db_session as session:
            session.add(user)
            assert user.organization.id.startswith('organization-'), "Organization id is prefixed incorrectly"
            assert str(user.organization._id) in user.organization.id, "Organization id is not using the correct uuid"