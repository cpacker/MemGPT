from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memgpt.log import get_logger
from memgpt.orm.mixins import UserMixin
from memgpt.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)


class Token(SqlalchemyBase, UserMixin):
    __tablename__ = "token"
    # __pydantic_model__ =  PydanticToken

    _temporary_shim_api_key: Mapped[Optional[str]] = mapped_column(
        String, default=lambda: "sk-" + str(uuid4()), doc="a temporary shim to get the ORM launched without refactoring downstream"
    )
    hash: Mapped[str] = mapped_column(String, doc="the secured one-way hash of the token")
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a name to identify the token")

    user: Mapped["User"] = relationship("User", back_populates="tokens")

    @hybrid_property
    def api_key(self) -> Optional[str]:
        """enforce read-only on temporary shim api key"""
        logger.warning(
            "Token.api_key is a temporary shim to get the ORM launched. It is unsecure, dangerous, and will be replaced by token.authenticate() in the next PR!"
        )
        return self._temporary_shim_api_key

    @classmethod
    def factory(cls, user_id: "UUID", name: Optional[str] = None) -> "Token":
        """Note: this is a temporary shim to get the ORM launched. It will immediately be replaced with proper
        secure token generation!
        """
        return cls(user_id=user_id, name=name)

    @classmethod
    def get_by_api_key(cls, db_session: "Session", api_key: str) -> "Token":
        """temporary lookup (insecure! replace!) to get a token by the plain text user api key"""
        return db_session.query(cls).filter(cls._temporary_shim_api_key == api_key, cls.is_deleted == False).one()
