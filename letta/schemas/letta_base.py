import uuid
from datetime import datetime
from logging import getLogger
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# from: https://gist.github.com/norton120/22242eadb80bf2cf1dd54a961b151c61


logger = getLogger(__name__)


class LettaBase(BaseModel):
    """Base schema for Letta schemas (does not include model provider schemas, e.g. OpenAI)"""

    model_config = ConfigDict(
        # allows you to use the snake or camelcase names in your code (ie user_id or userId)
        populate_by_name=True,
        # allows you do dump a sqlalchemy object directly (ie PersistedAddress.model_validate(SQLAdress)
        from_attributes=True,
        # throw errors if attributes are given that don't belong
        extra="forbid",
        # handle datetime serialization consistently across all models
        # json_encoders={datetime: lambda dt: (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).isoformat()},
    )

    # def __id_prefix__(self):
    #    raise NotImplementedError("All schemas must have an __id_prefix__ attribute!")

    @classmethod
    def generate_id_field(cls, prefix: Optional[str] = None) -> "Field":
        prefix = prefix or cls.__id_prefix__

        return Field(
            ...,
            description=cls._id_description(prefix),
            pattern=cls._id_regex_pattern(prefix),
            examples=[cls._id_example(prefix)],
            default_factory=cls._generate_id,
        )

    @classmethod
    def _generate_id(cls, prefix: Optional[str] = None) -> str:
        prefix = prefix or cls.__id_prefix__
        return f"{prefix}-{uuid.uuid4()}"

    # def _generate_id(self) -> str:
    #    return f"{self.__id_prefix__}-{uuid.uuid4()}"

    @classmethod
    def _id_regex_pattern(cls, prefix: str):
        """generates the regex pattern for a given id"""
        return (
            r"^" + prefix + r"-"  # prefix string
            r"[a-fA-F0-9]{8}"  # 8 hexadecimal characters
            # r"[a-fA-F0-9]{4}-"  # 4 hexadecimal characters
            # r"[a-fA-F0-9]{4}-"  # 4 hexadecimal characters
            # r"[a-fA-F0-9]{4}-"  # 4 hexadecimal characters
            # r"[a-fA-F0-9]{12}$"  # 12 hexadecimal characters
        )

    @classmethod
    def _id_example(cls, prefix: str):
        """generates an example id for a given prefix"""
        return f"{prefix}-123e4567-e89b-12d3-a456-426614174000"

    @classmethod
    def _id_description(cls, prefix: str):
        """generates a factory function for a given prefix"""
        return f"The human-friendly ID of the {prefix.capitalize()}"

    @field_validator("id", check_fields=False, mode="before")
    @classmethod
    def allow_bare_uuids(cls, v, values):
        """to ease the transition to stripe ids,
        we allow bare uuids and convert them with a warning
        """
        _ = values  # for SCA
        if isinstance(v, UUID):
            logger.debug(f"Bare UUIDs are deprecated, please use the full prefixed id ({cls.__id_prefix__})!")
            return f"{cls.__id_prefix__}-{v}"
        return v


class OrmMetadataBase(LettaBase):
    # metadata fields
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this object.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that made this object.")
    created_at: Optional[datetime] = Field(None, description="The timestamp when the object was created.")
    updated_at: Optional[datetime] = Field(None, description="The timestamp when the object was last updated.")
