from typing import TYPE_CHECKING, List, Literal, Optional, Type

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from letta.log import get_logger
from letta.orm.base import Base, CommonSqlalchemyMetaMixins
from letta.orm.errors import NoResultFound

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.orm import Session

    # from letta.orm.user import User

logger = get_logger(__name__)


class SqlalchemyBase(CommonSqlalchemyMetaMixins, Base):
    __abstract__ = True

    __order_by_default__ = "created_at"

    id: Mapped[str] = mapped_column(String, primary_key=True)

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
                query = query.where(cls.id > cursor)

            # Add a limit to the query if provided
            query = query.order_by(cls.id).limit(limit)

            # Handle soft deletes if the class has the 'is_deleted' attribute
            if hasattr(cls, "is_deleted"):
                query = query.where(cls.is_deleted == False)

            # Execute the query and return the results as a list of model instances
            return list(session.execute(query).scalars())

    @classmethod
    def read(
        cls,
        db_session: "Session",
        identifier: Optional[str] = None,
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
        # Start the query
        query = select(cls)
        # Collect query conditions for better error reporting
        query_conditions = []

        # If an identifier is provided, add it to the query conditions
        if identifier is not None:
            query = query.where(cls.id == identifier)
            query_conditions.append(f"id='{identifier}'")

        if kwargs:
            query = query.filter_by(**kwargs)
            query_conditions.append(", ".join(f"{key}='{value}'" for key, value in kwargs.items()))

        if actor:
            query = cls.apply_access_predicate(query, actor, access)
            query_conditions.append(f"access level in {access} for actor='{actor}'")

        if hasattr(cls, "is_deleted"):
            query = query.where(cls.is_deleted == False)
            query_conditions.append("is_deleted=False")
        if found := db_session.execute(query).scalar():
            return found

        # Construct a detailed error message based on query conditions
        conditions_str = ", ".join(query_conditions) if query_conditions else "no specific conditions"
        raise NoResultFound(f"{cls.__name__} not found with {conditions_str}")

    def create(self, db_session: "Session", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        with db_session as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            return self

    def delete(self, db_session: "Session", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True
        return self.update(db_session)

    def hard_delete(self, db_session: "Session", actor: Optional["User"] = None) -> None:
        """Permanently removes the record from the database."""
        if actor:
            logger.info(f"User {actor.id} requested hard deletion of {self.__class__.__name__} with ID {self.id}")

        with db_session as session:
            try:
                session.delete(self)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}")
                raise ValueError(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}: {e}")
            else:
                logger.info(f"{self.__class__.__name__} with ID {self.id} successfully hard deleted")

    def update(self, db_session: "Session", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

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
        org_id = getattr(actor, "organization_id", None)
        if not org_id:
            raise ValueError(f"object {actor} has no organization accessor")
        return query.where(cls.organization_id == org_id, cls.is_deleted == False)

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
