from uuid import uuid4, UUID
from typing import Optional, TYPE_CHECKING,Type, Union, List, Literal
from humps import depascalize
from sqlalchemy import select, UUID as SQLUUID
from sqlalchemy.orm import (
    Mapped,
    mapped_column
)
from memgpt.log import get_logger
from memgpt.orm.base import CommonSqlalchemyMetaMixins
from memgpt.orm.errors import NoResultFound

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from sqlalchemy import Select
    from memgpt.orm.user import User

logger = get_logger(__name__)


class SqlalchemyBase(CommonSqlalchemyMetaMixins):
    __abstract__ = True

    __order_by_default__ = "created_at"

    _id: Mapped[UUID] = mapped_column(SQLUUID(), primary_key=True, default=uuid4)

    @property
    def __prefix__(self) -> str:
        return depascalize(self.__class__.__name__)

    @property
    def id(self) -> Optional[str]:
        if self._id:
            return f"{self.__prefix__}-{self._id}"

    @id.setter
    def id(self, value: str) -> None:
        if not value:
            return
        prefix, id_ = value.split("-", 1)
        assert (
            prefix == self.__prefix__
        ), f"{prefix} is not a valid id prefix for {self.__class__.__name__}"
        self._id = UUID(id_)
    @classmethod
    def list(cls, db_session: "Session") -> list[Type["Base"]]:
        with db_session as session:
            return session.query(cls).all()

    @classmethod
    def read(
        cls, db_session: "Session", identifier: Union[str, UUID], **kwargs
    ) -> Type["SqlalchemyBase"]:
        del kwargs
        identifier = cls.to_uid(identifier)
        if found := db_session.get(cls, identifier):
            return found
        raise NoResultFound(f"{cls.__name__} with id {identifier} not found")

    def create(self, db_session: "Session") -> Type["SqlalchemyBase"]:
        with db_session as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            return self

    def delete(self, db_session: "Session") -> Type["SqlalchemyBase"]:
        self.deleted = True
        return self.update(db_session)

    def update(self, db_session: "Session") -> Type["SqlalchemyBase"]:
        with db_session as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            return self

    @classmethod
    def apply_access_predicate(
        cls,
        query: "Select",
        actor: "User",
        access: List[Literal["read", "write", "admin"]],
    ) -> "Select":
        """applies a WHERE clause restricting results to the given actor and access level
        Args:
            query: The initial sqlalchemy select statement
            actor: The user acting on the query. **Note**: this is called 'actor' to identify the
                   person or system acting. Users can act on users, making naming very sticky otherwise.
            access:
                what mode of access should the query restrict to? This will be used with granular permissions,
                but because of how it will impact every query we want to be explicitly calling access ahead of time.
        Returns:
            the sqlalchemy select statement restricted to the given access.
        """
        del access  # entrypoint for row-level permissions. Defaults to "same org as the actor, all permissions" at the moment
        org_uid = getattr(
            actor, "_organization_id", getattr(actor.organization, "_id", None)
        )
        if not org_uid:
            raise ValueError("object %s has no organization accessor", actor)
        return query.where(cls._organization_id == org_uid)