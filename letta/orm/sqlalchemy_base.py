from typing import TYPE_CHECKING, List, Literal, Optional, Type, Union
from uuid import UUID, uuid4

from humps import depascalize
from sqlalchemy import Boolean, String, select
from sqlalchemy.orm import Mapped, mapped_column

from letta.log import get_logger
from letta.orm.base import Base, CommonSqlalchemyMetaMixins
from letta.orm.errors import NoResultFound
from letta.orm.mixins import is_valid_uuid4

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.orm import Session

    # from letta.orm.user import User

logger = get_logger(__name__)


class SqlalchemyBase(CommonSqlalchemyMetaMixins, Base):
    __abstract__ = True

    __order_by_default__ = "created_at"

    _id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"{uuid4()}")

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, doc="Is this record deleted? Used for universal soft deletes.")

    @classmethod
    def __prefix__(cls) -> str:
        return depascalize(cls.__name__)

    @property
    def id(self) -> Optional[str]:
        if self._id:
            return f"{self.__prefix__()}-{self._id}"

    @id.setter
    def id(self, value: str) -> None:
        if not value:
            return
        prefix, id_ = value.split("-", 1)
        assert prefix == self.__prefix__(), f"{prefix} is not a valid id prefix for {self.__class__.__name__}"
        assert is_valid_uuid4(id_), f"{id_} is not a valid uuid4"
        self._id = id_

    @classmethod
    def list(
        cls, *, db_session: "Session", cursor: Optional[str] = None, limit: Optional[int] = 50, **kwargs
    ) -> List[Type["SqlalchemyBase"]]:
        """List records with optional cursor (for pagination) and limit."""
        with db_session as session:
            # Start with the base query filtered by kwargs
            query = select(cls).filter_by(**kwargs)

            # Add a cursor condition if provided
            if cursor:
                cursor_uuid = cls.get_uid_from_identifier(cursor)  # Assuming the cursor is an _id value
                query = query.where(cls._id > cursor_uuid)

            # Add a limit to the query if provided
            query = query.order_by(cls._id).limit(limit)

            # Handle soft deletes if the class has the 'is_deleted' attribute
            if hasattr(cls, "is_deleted"):
                query = query.where(cls.is_deleted == False)

            # Execute the query and return the results as a list of model instances
            return list(session.execute(query).scalars())

    @classmethod
    def get_uid_from_identifier(cls, identifier: str, indifferent: Optional[bool] = False) -> str:
        """converts the id into a uuid object
        Args:
            identifier: the string identifier, such as `organization-xxxx-xx...`
            indifferent: if True, will not enforce the prefix check
        """
        try:
            uuid_string = identifier.split("-", 1)[1] if indifferent else identifier.replace(f"{cls.__prefix__()}-", "")
            assert is_valid_uuid4(uuid_string)
            return uuid_string
        except ValueError as e:
            raise ValueError(f"{identifier} is not a valid identifier for class {cls.__name__}") from e

    @classmethod
    def read(
        cls,
        db_session: "Session",
        identifier: Union[str, UUID],
        actor: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        **kwargs,
    ) -> Type["SqlalchemyBase"]:
        """The primary accessor for an ORM record.
        Args:
            db_session: the database session to use when retrieving the record
            identifier: the identifier of the record to read, can be the id string or the UUID object for backwards compatibility
            actor: if specified, results will be scoped only to records the user is able to access
            access: if actor is specified, records will be filtered to the minimum permission level for the actor
            kwargs: additional arguments to pass to the read, used for more complex objects
        Returns:
            The matching object
        Raises:
            NoResultFound: if the object is not found
        """
        del kwargs  # arity for more complex reads
        identifier = cls.get_uid_from_identifier(identifier)
        query = select(cls).where(cls._id == identifier)
        # if actor:
        #     query = cls.apply_access_predicate(query, actor, access)
        if hasattr(cls, "is_deleted"):
            query = query.where(cls.is_deleted == False)
        if found := db_session.execute(query).scalar():
            return found
        raise NoResultFound(f"{cls.__name__} with id {identifier} not found")

    def create(self, db_session: "Session") -> Type["SqlalchemyBase"]:
        # self._infer_organization(db_session)

        with db_session as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            return self

    def delete(self, db_session: "Session") -> Type["SqlalchemyBase"]:
        self.is_deleted = True
        return self.update(db_session)

    def update(self, db_session: "Session") -> Type["SqlalchemyBase"]:
        with db_session as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            return self

    @classmethod
    def read_or_create(cls, *, db_session: "Session", **kwargs) -> Type["SqlalchemyBase"]:
        """get an instance by search criteria or create it if it doesn't exist"""
        try:
            return cls.read(db_session=db_session, identifier=kwargs.get("id", None))
        except NoResultFound:
            clean_kwargs = {k: v for k, v in kwargs.items() if k in cls.__table__.columns}
            return cls(**clean_kwargs).create(db_session=db_session)

    # TODO: Add back later when access predicates are actually important
    # The idea behind this is that you can add a WHERE clause restricting the actions you can take, e.g. R/W
    # @classmethod
    # def apply_access_predicate(
    #     cls,
    #     query: "Select",
    #     actor: "User",
    #     access: List[Literal["read", "write", "admin"]],
    # ) -> "Select":
    #     """applies a WHERE clause restricting results to the given actor and access level
    #     Args:
    #         query: The initial sqlalchemy select statement
    #         actor: The user acting on the query. **Note**: this is called 'actor' to identify the
    #                person or system acting. Users can act on users, making naming very sticky otherwise.
    #         access:
    #             what mode of access should the query restrict to? This will be used with granular permissions,
    #             but because of how it will impact every query we want to be explicitly calling access ahead of time.
    #     Returns:
    #         the sqlalchemy select statement restricted to the given access.
    #     """
    #     del access  # entrypoint for row-level permissions. Defaults to "same org as the actor, all permissions" at the moment
    #     org_uid = getattr(actor, "_organization_id", getattr(actor.organization, "_id", None))
    #     if not org_uid:
    #         raise ValueError("object %s has no organization accessor", actor)
    #     return query.where(cls._organization_id == org_uid, cls.is_deleted == False)

    @property
    def __pydantic_model__(self) -> Type["BaseModel"]:
        raise NotImplementedError("Sqlalchemy models must declare a __pydantic_model__ property to be convertable.")

    def to_pydantic(self) -> Type["BaseModel"]:
        """converts to the basic pydantic model counterpart"""
        return self.__pydantic_model__.model_validate(self)

    def to_record(self) -> Type["BaseModel"]:
        """Deprecated accessor for to_pydantic"""
        logger.warning("to_record is deprecated, use to_pydantic instead.")
        return self.to_pydantic()

    def _infer_organization(self, db_session: "Session") -> None:
        """🪄 MAGIC ALERT! 🪄
        Because so much of the original API is centered around user scopes,
        this allows us to continue with that scope and then infer the org from the creating user.

        IF a created_by_id is set, we will use that to infer the organization and magic set it at create time!
        If not do nothing to the object. Mutates in place.
        """
        if self.created_by_id and hasattr(self, "_organization_id"):
            try:
                from letta.orm.user import User  # to avoid circular import

                created_by = User.read(db_session, self.created_by_id)
            except NoResultFound:
                logger.warning(f"User {self.created_by_id} not found, unable to infer organization.")
                return
            self._organization_id = created_by._organization_id
