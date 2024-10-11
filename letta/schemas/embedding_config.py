from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """

    Embedding model configuration. This object specifies all the information necessary to access an embedding model to usage with Letta, except for secret keys.

    Attributes:
        embedding_endpoint_type (str): The endpoint type for the model.
        embedding_endpoint (str): The endpoint for the model.
        embedding_model (str): The model for the embedding.
        embedding_dim (int): The dimension of the embedding.
        embedding_chunk_size (int): The chunk size of the embedding.
        azure_endpoint (:obj:`str`, optional): The Azure endpoint for the model (Azure only).
        azure_version (str): The Azure version for the model (Azure only).
        azure_deployment (str): The Azure deployment for the model (Azure only).

    """

    embedding_endpoint_type: str = Field(..., description="The endpoint type for the model.")
    embedding_endpoint: Optional[str] = Field(None, description="The endpoint for the model (`None` if local).")
    embedding_model: str = Field(..., description="The model for the embedding.")
    embedding_dim: int = Field(..., description="The dimension of the embedding.")
    embedding_chunk_size: Optional[int] = Field(300, description="The chunk size of the embedding.")

    # azure only
    azure_endpoint: Optional[str] = Field(None, description="The Azure endpoint for the model.")
    azure_version: Optional[str] = Field(None, description="The Azure version for the model.")
    azure_deployment: Optional[str] = Field(None, description="The Azure deployment for the model.")

    @classmethod
    def default_config(cls, model_name: Optional[str] = None, provider: Optional[str] = None):

        if model_name == "text-embedding-ada-002" or (not model_name and provider == "openai"):
            return cls(
                embedding_model="text-embedding-ada-002",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=300,
            )
        elif model_name == "letta":
            return cls(
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_dim=1024,
                embedding_chunk_size=300,
                embedding_endpoint_type="hugging-face",
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def pretty_print(self) -> str:
        return (
            f"{self.embedding_model}"
            + (f" [type={self.embedding_endpoint_type}]" if self.embedding_endpoint_type else "")
            + (f" [ip={self.embedding_endpoint}]" if self.embedding_endpoint else "")
        )
