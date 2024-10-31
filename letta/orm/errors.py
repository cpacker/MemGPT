class NoResultFound(Exception):
    """A record or records cannot be found given the provided search params"""


class MalformedIdError(Exception):
    """An id not in the right format, most likely violating uuid4 format."""
