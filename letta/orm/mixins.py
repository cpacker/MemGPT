from typing import Optional, Type
from uuid import UUID

from letta.orm.base import Base


class MalformedIdError(Exception):
    pass


def _relation_getter(instance: "Base", prop: str) -> Optional[str]:
    prefix = prop.replace("_", "")
    formatted_prop = f"_{prop}_id"
    try:
        uuid_ = getattr(instance, formatted_prop)
        return f"{prefix}-{uuid_}"
    except AttributeError:
        return None


def _relation_setter(instance: Type["Base"], prop: str, value: str) -> None:
    formatted_prop = f"_{prop}_id"
    prefix = prop.replace("_", "")
    if not value:
        setattr(instance, formatted_prop, None)
        return
    try:
        found_prefix, id_ = value.split("-", 1)
    except ValueError as e:
        raise MalformedIdError(f"{value} is not a valid ID.") from e
    assert (
        # TODO: should be able to get this from the Mapped typing, not sure how though
        # prefix = getattr(?, "prefix")
        found_prefix
        == prefix
    ), f"{found_prefix} is not a valid id prefix, expecting {prefix}"
    try:
        setattr(instance, formatted_prop, UUID(id_))
    except ValueError as e:
        raise MalformedIdError("Hash segment of {value} is not a valid UUID") from e
