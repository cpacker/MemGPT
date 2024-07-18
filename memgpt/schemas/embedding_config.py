from typing import Optional

from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    embedding_endpoint_type: Optional[str] = "openai"
    embedding_endpoint: Optional[str] = "https://api.openai.com/v1"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    embedding_dim: Optional[int] = 1536
    embedding_chunk_size: Optional[int] = 300


class OpenAIEmbeddingConfig(EmbeddingConfig):
    def __init__(self, openai_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.openai_key = openai_key


class AzureEmbeddingConfig(EmbeddingConfig):
    def __init__(
        self,
        azure_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.azure_key = azure_key
        self.azure_endpoint = azure_endpoint
        self.azure_version = azure_version
        self.azure_deployment = azure_deployment
