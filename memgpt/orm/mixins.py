from typing import Optional, Type
from uuid import UUID

from sqlalchemy import UUID as SQLUUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.base import Base


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


class OrganizationMixin(Base):
    """Mixin for models that belong to an organization."""

    __abstract__ = True

    _organization_id: Mapped[UUID] = mapped_column(SQLUUID(), ForeignKey("organization._id"))

    @property
    def organization_id(self) -> str:
        return _relation_getter(self, "organization")

    @organization_id.setter
    def organization_id(self, value: str) -> None:
        _relation_setter(self, "organization", value)


class UserMixin(Base):
    """Mixin for models that belong to a user."""

    __abstract__ = True

    _user_id: Mapped[UUID] = mapped_column(SQLUUID(), ForeignKey("user._id"))

    @property
    def user_id(self) -> str:
        return _relation_getter(self, "user")

    @user_id.setter
    def user_id(self, value: str) -> None:
        _relation_setter(self, "user", value)


class AgentMixin(Base):
    """Mixin for models that belong to an agent."""

    __abstract__ = True

    _agent_id: Mapped[UUID] = mapped_column(SQLUUID(), ForeignKey("agent._id"))

    @property
    def agent_id(self) -> str:
        return _relation_getter(self, "agent")

    @agent_id.setter
    def agent_id(self, value: str) -> None:
        _relation_setter(self, "agent", value)


class DocumentMixin(Base):
    """Mixin for models that belong to a document."""

    __abstract__ = True

    _document_id: Mapped[Optional[UUID]] = mapped_column(SQLUUID(), ForeignKey("document._id"))

    @property
    def document_id(self) -> str:
        return _relation_getter(self, "document")

    @document_id.setter
    def document_id(self, value: str) -> None:
        _relation_setter(self, "document", value)


class HumanMemoryTemplateMixin(Base):
    """Mixin for models that have one human memory template relationships."""

    __abstract__ = True

    _human_memory_template_id: Mapped[UUID] = mapped_column(SQLUUID(), ForeignKey("memory_template._id"))

    @property
    def human_memory_template_id(self) -> str:
        return _relation_getter(self, "human_memory_template")

    @human_memory_template_id.setter
    def human_memory_template_id(self, value: str) -> None:
        _relation_setter(self, "human_memory_template", value)


class SystemMemoryTemplateMixin(Base):
    """Mixin for models that have one system memory template relationships."""

    __abstract__ = True

    _system_memory_template_id: Mapped[UUID] = mapped_column(SQLUUID(), ForeignKey("memory_template._id"))

    @property
    def system_memory_template_id(self) -> str:
        return _relation_getter(self, "system_memory_template")

    @system_memory_template_id.setter
    def system_memory_template_id(self, value: str) -> None:
        _relation_setter(self, "system_memory_template", value)


class PersonaMemoryTemplateMixin(Base):
    """Mixin for models that have one persona memory template relationships."""

    __abstract__ = True

    _persona_memory_template_id: Mapped[UUID] = mapped_column(SQLUUID(), ForeignKey("memory_template._id"))

    @property
    def persona_memory_template_id(self) -> str:
        return _relation_getter(self, "persona_memory_template")

    @persona_memory_template_id.setter
    def persona_memory_template_id(self, value: str) -> None:
        _relation_setter(self, "persona_memory_template", value)
