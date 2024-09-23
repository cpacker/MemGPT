from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class BaseAPIKey(LettaBase):
    __id_prefix__ = "sk"  # secret key


class APIKey(BaseAPIKey):
    id: str = BaseAPIKey.generate_id_field()
    user_id: str = Field(..., description="The unique identifier of the user associated with the token.")
    key: str = Field(..., description="The key value.")
    name: str = Field(..., description="Name of the token.")


class APIKeyCreate(BaseAPIKey):
    user_id: str = Field(..., description="The unique identifier of the user associated with the token.")
    name: Optional[str] = Field(None, description="Name of the token.")
