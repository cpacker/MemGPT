class NoResultFound(Exception):
    """A record or records cannot be found given the provided search params"""


class MalformedIdError(Exception):
    """An id not in the right format, most likely violating uuid4 format."""


class UniqueConstraintViolationError(ValueError):
    """Custom exception for unique constraint violations."""


class ForeignKeyConstraintViolationError(ValueError):
    """Custom exception for foreign key constraint violations."""


class DatabaseTimeoutError(Exception):
    """Custom exception for database timeout issues."""

    def __init__(self, message="Database operation timed out", original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
