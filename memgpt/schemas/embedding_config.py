from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """

    Embedding model configuration. This object specifies all the information necessary to access an embedding model to usage with MemGPT, except for secret keys.

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
