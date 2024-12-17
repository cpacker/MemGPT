from datetime import datetime
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, List, Literal, Optional

from sqlalchemy import String, desc, func, or_, select
from sqlalchemy.exc import DBAPIError, IntegrityError, TimeoutError
from sqlalchemy.orm import Mapped, Session, mapped_column

from letta.log import get_logger
from letta.orm.base import Base, CommonSqlalchemyMetaMixins
from letta.orm.errors import (
    DatabaseTimeoutError,
    ForeignKeyConstraintViolationError,
    NoResultFound,
    UniqueConstraintViolationError,
)
from letta.orm.sqlite_functions import adapt_array

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.orm import Session


logger = get_logger(__name__)


def handle_db_timeout(func):
    """Decorator to handle SQLAlchemy TimeoutError and wrap it in a custom exception."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TimeoutError as e:
            logger.error(f"Timeout while executing {func.__name__} with args {args} and kwargs {kwargs}: {e}")
            raise DatabaseTimeoutError(message=f"Timeout occurred in {func.__name__}.", original_exception=e)

    return wrapper


class AccessType(str, Enum):
    ORGANIZATION = "organization"
    USER = "user"


class SqlalchemyBase(CommonSqlalchemyMetaMixins, Base):
    __abstract__ = True

    __order_by_default__ = "created_at"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    @classmethod
    @handle_db_timeout
    def list(
        cls,
        *,
        db_session: "Session",
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        ascending: bool = True,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
        **kwargs,
    ) -> List["SqlalchemyBase"]:
        """
        List records with cursor-based pagination, ordering by created_at.
        Cursor is an ID, but pagination is based on the cursor object's created_at value.

        Args:
            db_session: SQLAlchemy session
            cursor: ID of the last item seen (for pagination)
            start_date: Filter items after this date
            end_date: Filter items before this date
            limit: Maximum number of items to return
            query_text: Text to search for
            query_embedding: Vector to search for similar embeddings
            ascending: Sort direction
            tags: List of tags to filter by
            match_all_tags: If True, return items matching all tags. If False, match any tag.
            **kwargs: Additional filters to apply
        """
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be earlier than or equal to end_date")

        logger.debug(f"Listing {cls.__name__} with kwarg filters {kwargs}")
        with db_session as session:
            # If cursor provided, get the reference object
            cursor_obj = None
            if cursor:
                cursor_obj = session.get(cls, cursor)
                if not cursor_obj:
                    raise NoResultFound(f"No {cls.__name__} found with id {cursor}")

            query = select(cls)

            # Handle tag filtering if the model has tags
            if tags and hasattr(cls, "tags"):
                query = select(cls)

                if match_all_tags:
                    # Match ALL tags - use subqueries
                    for tag in tags:
                        subquery = select(cls.tags.property.mapper.class_.agent_id).where(cls.tags.property.mapper.class_.tag == tag)
                        query = query.filter(cls.id.in_(subquery))
                else:
                    # Match ANY tag - use join and filter
                    query = (
                        query.join(cls.tags).filter(cls.tags.property.mapper.class_.tag.in_(tags)).group_by(cls.id)  # Deduplicate results
                    )

                # Group by primary key and all necessary columns to avoid JSON comparison
                query = query.group_by(cls.id)

            # Apply filtering logic from kwargs
            for key, value in kwargs.items():
                column = getattr(cls, key)
                if isinstance(value, (list, tuple, set)):
                    query = query.where(column.in_(value))
                else:
                    query = query.where(column == value)

            # Date range filtering
            if start_date:
                query = query.filter(cls.created_at > start_date)
            if end_date:
                query = query.filter(cls.created_at < end_date)

            # Cursor-based pagination
            if cursor_obj:
                if ascending:
                    query = query.where(cls.created_at >= cursor_obj.created_at).where(
                        or_(cls.created_at > cursor_obj.created_at, cls.id > cursor_obj.id)
                    )
                else:
                    query = query.where(cls.created_at <= cursor_obj.created_at).where(
                        or_(cls.created_at < cursor_obj.created_at, cls.id < cursor_obj.id)
                    )

            # Text search
            if query_text:
                query = query.filter(func.lower(cls.text).contains(func.lower(query_text)))

            # Embedding search (for Passages)
            is_ordered = False
            if query_embedding:
                if not hasattr(cls, "embedding"):
                    raise ValueError(f"Class {cls.__name__} does not have an embedding column")

                from letta.settings import settings

                if settings.letta_pg_uri_no_default:
                    # PostgreSQL with pgvector
                    query = query.order_by(cls.embedding.cosine_distance(query_embedding).asc())
                else:
                    # SQLite with custom vector type
                    query_embedding_binary = adapt_array(query_embedding)
                    query = query.order_by(
                        func.cosine_distance(cls.embedding, query_embedding_binary).asc(), cls.created_at.asc(), cls.id.asc()
                    )
                    is_ordered = True

            # Handle soft deletes
            if hasattr(cls, "is_deleted"):
                query = query.where(cls.is_deleted == False)

            # Apply ordering
            if not is_ordered:
                if ascending:
                    query = query.order_by(cls.created_at, cls.id)
                else:
                    query = query.order_by(desc(cls.created_at), desc(cls.id))

            query = query.limit(limit)

            return list(session.execute(query).scalars())

    @classmethod
    @handle_db_timeout
    def read(
        cls,
        db_session: "Session",
        identifier: Optional[str] = None,
        actor: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        **kwargs,
    ) -> "SqlalchemyBase":
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

    @handle_db_timeout
    def create(self, db_session: "Session", actor: Optional["User"] = None) -> "SqlalchemyBase":
        logger.debug(f"Creating {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        if actor:
            self._set_created_and_updated_by_fields(actor.id)
        try:
            with db_session as session:
                session.add(self)
                session.commit()
                session.refresh(self)
                return self
        except (DBAPIError, IntegrityError) as e:
            self._handle_dbapi_error(e)

    @handle_db_timeout
    def delete(self, db_session: "Session", actor: Optional["User"] = None) -> "SqlalchemyBase":
        logger.debug(f"Soft deleting {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True
        return self.update(db_session)

    @handle_db_timeout
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
                logger.debug(f"{self.__class__.__name__} with ID {self.id} successfully hard deleted")

    @handle_db_timeout
    def update(self, db_session: "Session", actor: Optional["User"] = None) -> "SqlalchemyBase":
        logger.debug(f"Updating {self.__class__.__name__} with ID: {self.id} with actor={actor}")
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        with db_session as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            return self

    @classmethod
    @handle_db_timeout
    def size(
        cls,
        *,
        db_session: "Session",
        actor: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        **kwargs,
    ) -> int:
        """
        Get the count of rows that match the provided filters.

        Args:
            db_session: SQLAlchemy session
            **kwargs: Filters to apply to the query (e.g., column_name=value)

        Returns:
            int: The count of rows that match the filters

        Raises:
            DBAPIError: If a database error occurs
        """
        logger.debug(f"Calculating size for {cls.__name__} with filters {kwargs}")

        with db_session as session:
            query = select(func.count()).select_from(cls)

            if actor:
                query = cls.apply_access_predicate(query, actor, access, access_type)

            # Apply filtering logic based on kwargs
            for key, value in kwargs.items():
                if value:
                    column = getattr(cls, key, None)
                    if not column:
                        raise AttributeError(f"{cls.__name__} has no attribute '{key}'")
                    if isinstance(value, (list, tuple, set)):  # Check for iterables
                        query = query.where(column.in_(value))
                    else:  # Single value for equality filtering
                        query = query.where(column == value)

            # Handle soft deletes if the class has the 'is_deleted' attribute
            if hasattr(cls, "is_deleted"):
                query = query.where(cls.is_deleted == False)

            try:
                count = session.execute(query).scalar()
                return count if count else 0
            except DBAPIError as e:
                logger.exception(f"Failed to calculate size for {cls.__name__}")
                raise e

    @classmethod
    def apply_access_predicate(
        cls,
        query: "Select",
        actor: "User",
        access: List[Literal["read", "write", "admin"]],
        access_type: AccessType = AccessType.ORGANIZATION,
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
        if access_type == AccessType.ORGANIZATION:
            org_id = getattr(actor, "organization_id", None)
            if not org_id:
                raise ValueError(f"object {actor} has no organization accessor")
            return query.where(cls.organization_id == org_id, cls.is_deleted == False)
        elif access_type == AccessType.USER:
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
    def __pydantic_model__(self) -> "BaseModel":
        raise NotImplementedError("Sqlalchemy models must declare a __pydantic_model__ property to be convertable.")

    def to_pydantic(self) -> "BaseModel":
        """converts to the basic pydantic model counterpart"""
        return self.__pydantic_model__.model_validate(self)

    def to_record(self) -> "BaseModel":
        """Deprecated accessor for to_pydantic"""
        logger.warning("to_record is deprecated, use to_pydantic instead.")
        return self.to_pydantic()
