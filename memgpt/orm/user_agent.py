from sqlachemy.orm import UniqueConstraint


from memgpt.orm.base import Base
from memgpt.orm.mixins import UserMixin, AgentMixin

class UserAgent(Base, UserMixin, AgentMixin):
    __tablename__ = 'user_agent'
    __table_args__ = (
        UniqueConstraint(
            "_agent_id",
            "_user_id",
            name="unique_agent_user_constraint",
        ),
    )