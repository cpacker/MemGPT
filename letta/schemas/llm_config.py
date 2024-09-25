from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    """
    Configuration for a Language Model (LLM) model. This object specifies all the information necessary to access an LLM model to usage with Letta, except for secret keys.

    Attributes:
        model (str): The name of the LLM model.
        model_endpoint_type (str): The endpoint type for the model.
        model_endpoint (str): The endpoint for the model.
        model_wrapper (str): The wrapper for the model.
        context_window (int): The context window size for the model.
    """

    # TODO: 🤮 don't default to a vendor! bug city!
    model: str = Field(..., description="LLM model name. ")
    model_endpoint_type: str = Field(..., description="The endpoint type for the model.")
    model_endpoint: str = Field(..., description="The endpoint for the model.")
    model_wrapper: Optional[str] = Field(None, description="The wrapper for the model.")
    context_window: int = Field(..., description="The context window size for the model.")

    # FIXME hack to silence pydantic protected namespace warning
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def default_config(cls, model_name: str):
        if model_name == "gpt-4":
            return cls(
                model="gpt-4",
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model_wrapper=None,
                context_window=8192,
            )
        elif model_name == "gpt-4o-mini":
            return cls(
                model="gpt-4o-mini",
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model_wrapper=None,
                context_window=128000,
            )
        elif model_name == "letta":
            return cls(
                model="memgpt-openai",
                model_endpoint_type="openai",
                model_endpoint="https://inference.memgpt.ai",
                context_window=16384,
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")
