from pytest import mark as m, raises
from memgpt.orm.errors import NoResultFound

from tests.mock_factory.models import (
    Agent,
    MockAgentFactory,
    MockOrganizationFactory,
    MockTokenFactory,
    MockToolFactory,
    MockUserFactory,
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
        ],
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

        assert user.id.startswith("user-")
        assert str(user._id) in user.id

        with db_session as session:
            session.add(user)
            assert user.organization.id.startswith("organization-"), "Organization id is prefixed incorrectly"
            assert str(user.organization._id) in user.organization.id, "Organization id is not using the correct uuid"

    @m.context("and retrieving an Agent model")
    @m.it("should respect the access predicate")
    def test_access_predicate_basic(self, db_session):
        star_wars = MockOrganizationFactory(db_session=db_session, name="star_wars").generate()
        star_trek = MockOrganizationFactory(db_session=db_session, name="star_trek").generate()
        luke = MockUserFactory(db_session=db_session, organization=star_wars).generate()
        schema_luke = luke.to_pydantic()

        spock = MockUserFactory(db_session=db_session, organization=star_trek).generate()
        schema_spock = spock.to_pydantic()

        c3po = MockAgentFactory(db_session=db_session, organization=star_wars).generate()
        r2d2 = MockAgentFactory(db_session=db_session, organization=star_wars).generate()
        data = MockAgentFactory(db_session=db_session, organization=star_trek).generate()


        _ = Agent.read(identifier=data.id, db_session=db_session, actor=spock, access=["read"])
        _ = Agent.read(identifier=data.id, db_session=db_session, actor=schema_spock, access=["read"])

        for droid in (c3po, r2d2,):
            _ = Agent.read(identifier=droid.id, db_session=db_session, actor=luke, access=["read"])
            _ = Agent.read(identifier=droid.id, db_session=db_session, actor=schema_luke, access=["read"])
            with raises(NoResultFound):
                _ = Agent.read(identifier=droid.id, db_session=db_session, actor=spock, access=["read"])
                _ = Agent.read(identifier=droid.id, db_session=db_session, actor=schema_spock, access=["read"])
                _ = Agent.read(identifier=data.id, db_session=db_session, actor=luke, access=["read"])
                _ = Agent.read(identifier=data.id, db_session=db_session, actor=schema_luke, access=["read"])