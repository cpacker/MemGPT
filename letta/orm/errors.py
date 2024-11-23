class NoResultFound(Exception):
    """A record or records cannot be found given the provided search params"""


class MalformedIdError(Exception):
    """An id not in the right format, most likely violating uuid4 format."""


class UniqueConstraintViolationError(ValueError):
    """Custom exception for unique constraint violations."""


class ForeignKeyConstraintViolationError(ValueError):
    """Custom exception for foreign key constraint violations."""
