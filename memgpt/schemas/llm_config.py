from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    # TODO: 🤮 don't default to a vendor! bug city!
    model: str = Field(..., description="LLM model name. ")
    model_endpoint_type: str = Field(..., description="The endpoint type for the model.")
    model_endpoint: str = Field(..., description="The endpoint for the model.")
    model_wrapper: Optional[str] = Field(None, description="The wrapper for the model.")
    context_window: int = Field(..., description="The context window size for the model.")

    # FIXME hack to silence pydantic protected namespace warning
    model_config = ConfigDict(protected_namespaces=())
