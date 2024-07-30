from typing import TYPE_CHECKING, Any
from uuid import uuid4
from copy import deepcopy
from faker import Faker

from memgpt.orm.organization import Organization
from memgpt.orm.agent import Agent
from memgpt.orm.user import User
from memgpt.orm.tool import Tool
from memgpt.orm.source import Source
from memgpt.orm.token import Token

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


faker = Faker()


class BaseMockFactory:
    __model__: "Any"
    db_session: "Session"
    model_dict: dict
    default_dict: dict

    def __init__(self, db_session: "Session", **kwargs):
        self.db_session = db_session
        self.model_dict = deepcopy(self.default_dict)
        self.model_dict.update(**kwargs)

    def generate(self) -> "Any":
        obj_ = self.__model__(**self.model_dict)
        with self.db_session as session:
            session.add(obj_)
            session.commit()
            session.refresh(obj_)
        return obj_


class MockOrganizationFactory(BaseMockFactory):
    __model__ = Organization

    def __init__(self, db_session: "Session", **kwargs):
        self.default_dict = {
            "name": faker.company(),
        }
        super().__init__(db_session, **kwargs)


class MockUserFactory(BaseMockFactory):
    __model__ = User

    def __init__(self, db_session: "Session", **kwargs):
        self.default_dict = {
            "name": faker.name(),
            "email": faker.email(),
        }
        super().__init__(db_session, **kwargs)

    def generate(self) -> "Any":
        if "organization_id" not in self.model_dict:
            org = MockOrganizationFactory(self.db_session).generate()
            self.model_dict["organization_id"] = org.id

        return super().generate()


class MockAgentFactory(BaseMockFactory):
    __model__ = Agent

    def __init__(self, db_session: "Session", **kwargs):
        self.default_dict = {
            "name": faker.name(),
        }
        super().__init__(db_session, **kwargs)

    def generate(self) -> "Any":
        if "organization_id" not in self.model_dict:
            org = MockOrganizationFactory(self.db_session).generate()
            self.model_dict["organization_id"] = org.id
        if "user_id" not in self.model_dict:
            user = MockUserFactory(model_dict=self.model_dict, db_session=self.db_session).generate()
            self.model_dict["user_id"] = user.id

        return super().generate()


class MockToolFactory(BaseMockFactory):
    __model__ = Tool

    def __init__(self, db_session: "Session", **kwargs):
        self.default_dict = {
            "name": "Python Tool",
            "source_type": "python",
            "source_code": "print('Hello World')",
        }
        super().__init__(db_session, **kwargs)

    def generate(self) -> "Any":
        if "organization_id" not in self.model_dict:
            org = MockOrganizationFactory(self.db_session).generate()
            self.model_dict["organization_id"] = org.id

        return super().generate()


class MockSourceFactory(BaseMockFactory):
    __model__ = Source

    def __init__(self, db_session: "Session", **kwargs):
        self.default_dict = {
            "name": faker.name(),
            "embedding_dim": 768,
            "embedding_model": faker.name(),
        }
        super().__init__(db_session, **kwargs)

    def generate(self) -> "Any":
        if "organization_id" not in self.model_dict:
            org = MockOrganizationFactory(self.db_session).generate()
            self.model_dict["organization_id"] = org.id
        if "agent_id" not in self.model_dict:
            agent = MockAgentFactory(self.db_session).generate()
            self.model_dict["agent_id"] = agent.id

        return super().generate()


class MockTokenFactory(BaseMockFactory):
    __model__ = Token

    def __init__(self, db_session: "Session", **kwargs):
        self.default_dict = {
            "name": faker.name(),
            "hash": str(uuid4()),
        }
        super().__init__(db_session, **kwargs)

    def generate(self) -> "Any":
        if "user_id" not in self.model_dict:
            user = MockUserFactory(model_dict=self.model_dict, db_session=self.db_session).generate()
            self.model_dict["user_id"] = user.id

        return super().generate()