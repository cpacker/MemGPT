from typing import Optional

from pydantic import BaseModel, ConfigDict


class LLMConfig(BaseModel):
    # TODO: 🤮 don't default to a vendor! bug city!
    model: Optional[str] = "gpt-4"
    model_endpoint_type: Optional[str] = "openai"
    model_endpoint: Optional[str] = "https://api.openai.com/v1"
    model_wrapper: Optional[str] = None
    context_window: Optional[int] = None

    # FIXME hack to silence pydantic protected namespace warning
    model_config = ConfigDict(protected_namespaces=())
