from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    config: dict = Field(..., description="The server configuration object.")
    defaults: dict = Field(..., description="The defaults for the configuration.")
