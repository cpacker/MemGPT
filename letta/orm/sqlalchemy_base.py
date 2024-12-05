from typing import TYPE_CHECKING, List, Literal, Optional, Type, Union, Tuple
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Boolean, func, asc, desc, select, or_, and_
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, Mapped, mapped_column
from sqlalchemy.sql import or_, and_

from letta.log import get_logger
from letta.orm.base import Base, CommonSqlalchemyMetaMixins
from letta.orm.errors import (
    ForeignKeyConstraintViolationError,
    NoResultFound,
    UniqueConstraintViolationError,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.orm import Session


logger = get_logger(__name__)


class SqlalchemyBase(CommonSqlalchemyMetaMixins, Base):
    __abstract__ = True

    __order_by_default__ = "created_at"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    @classmethod
    def get(cls, *, db_session: Session, id: str) -> Optional["SqlalchemyBase"]:
        """Get a record by ID.
        
        Args:
            db_session: SQLAlchemy session
            id: Record ID to retrieve
            
        Returns:
            Optional[SqlalchemyBase]: The record if found, None otherwise
        """
        try:
            return db_session.query(cls).filter(cls.id == id).first()
        except DBAPIError:
            return None

    @classmethod
    def list(
        cls,
        *,
        db_session: "Session",
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        **kwargs
    ) -> Union[List[Type["SqlalchemyBase"]], Tuple[Optional[str], List[Type["SqlalchemyBase"]]]]:
        """List records with advanced filtering and pagination options.
        
        Args:
            db_session: SQLAlchemy session
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            query_text: Optional text to search for in message content
            **kwargs: Additional filters to apply
            
        Returns:
            If using cursor-based pagination (after/before): Tuple[Optional[str], List[records]]
            Otherwise: List[records]
        """
        logger.debug(f"Listing {cls.__name__} with kwarg filters {kwargs}")
        with db_session as session:
            query = select(cls).filter_by(**kwargs)

            # Handle date range filtering
            if start_date and end_date and hasattr(cls, "created_at"):
                # If start_date equals end_date, add a small buffer to include records created at that exact time
                if start_date.date() == end_date.date():
                    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                query = query.filter(cls.created_at >= start_date)
                query = query.filter(cls.created_at <= end_date)
            elif start_date and hasattr(cls, "created_at"):
                query = query.filter(cls.created_at >= start_date)
            elif end_date and hasattr(cls, "created_at"):
                query = query.filter(cls.created_at <= end_date)

            # Handle role filters if the model has them
            if hasattr(cls, "role"):
                query = query.filter(cls.role != "system")
                query = query.filter(cls.role != "tool")
                
            # Handle cursor-based pagination
            if cursor:
                query = query.where(cls.id > cursor)
            
            # Apply text search
            if query_text:
                query = query.filter(func.lower(cls.text).contains(func.lower(query_text)))

            # Handle ordering (defaults are ascending, "created_at")
            # Priorities:
            #   1. cursor-based pagination
            #   2. Date
            query = query.order_by(cls.id).order_by(asc(cls.created_at)).limit(limit)

            # Handle soft deletes if the class has the 'is_deleted' attribute
            if hasattr(cls, "is_deleted"):
                query = query.where(cls.is_deleted == False)

            return list(session.execute(query).scalars())

    @classmethod
    def read(
        cls,
        db_session: "Session",
        identifier: Optional[str] = None,
        actor: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: str = "organization",
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
        logger.debug(f"Reading {cls.__name__} with ID: {identifier} with actor={actor}")

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
            query = cls.apply_access_predicate(query, actor, access, access_type)
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
        logger.debug(f"Creating {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        if actor:
            self._set_created_and_updated_by_fields(actor.id)
        try:
            with db_session as session:
                session.add(self)
                session.commit()
                session.refresh(self)
                return self
        except DBAPIError as e:
            self._handle_dbapi_error(e)

    def delete(self, db_session: "Session", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        logger.debug(f"Soft deleting {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True
        return self.update(db_session)

    def hard_delete(self, db_session: "Session", actor: Optional["User"] = None) -> None:
        """Permanently removes the record from the database."""
        logger.debug(f"Hard deleting {self.__class__.__name__} with ID: {self.id} with actor={actor}")

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
        logger.debug(f"Updating {self.__class__.__name__} with ID: {self.id} with actor={actor}")
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
        access_type: str = "organization",
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
        if access_type == "organization":
            org_id = getattr(actor, "organization_id", None)
            if not org_id:
                raise ValueError(f"object {actor} has no organization accessor")
            return query.where(cls.organization_id == org_id, cls.is_deleted == False)
        elif access_type == "user":
            user_id = getattr(actor, "id", None)
            if not user_id:
                raise ValueError(f"object {actor} has no user accessor")
            return query.where(cls.user_id == user_id, cls.is_deleted == False)
        else:
            raise ValueError(f"unknown access_type: {access_type}")

    @classmethod
    def _handle_dbapi_error(cls, e: DBAPIError):
        """Handle database errors and raise appropriate custom exceptions."""
        orig = e.orig  # Extract the original error from the DBAPIError
        error_code = None
        error_message = str(orig) if orig else str(e)
        logger.info(f"Handling DBAPIError: {error_message}")

        # Handle SQLite-specific errors
        if "UNIQUE constraint failed" in error_message:
            raise UniqueConstraintViolationError(
                f"A unique constraint was violated for {cls.__name__}. Check your input for duplicates: {e}"
            ) from e

        if "FOREIGN KEY constraint failed" in error_message:
            raise ForeignKeyConstraintViolationError(
                f"A foreign key constraint was violated for {cls.__name__}. Check your input for missing or invalid references: {e}"
            ) from e

        # For psycopg2
        if hasattr(orig, "pgcode"):
            error_code = orig.pgcode
        # For pg8000
        elif hasattr(orig, "args") and len(orig.args) > 0:
            # The first argument contains the error details as a dictionary
            err_dict = orig.args[0]
            if isinstance(err_dict, dict):
                error_code = err_dict.get("C")  # 'C' is the error code field
        logger.info(f"Extracted error_code: {error_code}")

        # Handle unique constraint violations
        if error_code == "23505":
            raise UniqueConstraintViolationError(
                f"A unique constraint was violated for {cls.__name__}. Check your input for duplicates: {e}"
            ) from e

        # Handle foreign key violations
        if error_code == "23503":
            raise ForeignKeyConstraintViolationError(
                f"A foreign key constraint was violated for {cls.__name__}. Check your input for missing or invalid references: {e}"
            ) from e

        # Re-raise for other unhandled DBAPI errors
        raise

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
