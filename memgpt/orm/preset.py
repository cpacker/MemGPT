from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from memgpt.orm.sqlalchemy_base import SqlalchemyBase
from memgpt.orm.mixins import OrganizationMixin

class Preset(SqlalchemyBase, OrganizationMixin):

    __tablename__ = 'preset'
    __table_args__ = (
        UniqueConstraint(
            "_organization_id",
            "name",
            name="unique_name_organization",
        ),
    )

    name: Mapped[str] = mapped_column(doc="the name of the preset, must be unique within the org", nullable=False)
    description: Mapped[str] = mapped_column(nullable=True, doc="a human-readable description of the preset")

    ## TODO: these are unclear - human vs human_name for example, what and why?
    system = Column(String)
    human = Column(String)
    human_name = Column(String, nullable=False)
    persona = Column(String)
    persona_name = Column(String, nullable=False)
    ## TODO: What is this?
    preset = Column(String)

    functions_schema = Column(JSON)