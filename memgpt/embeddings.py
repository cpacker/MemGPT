import typer
from typing import Optional, List
import os
from llama_index.embeddings import OpenAIEmbedding, AzureOpenAIEmbedding
from llama_index.embeddings import TextEmbeddingsInference
from llama_index.bridge.pydantic import PrivateAttr

from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_utils import format_query, format_text


class EmbeddingEndpoint(BaseEmbedding):

    """Implementation for OpenAI compatible endpoint"""

    """ Based off llama index https://github.com/run-llama/llama_index/blob/a98bdb8ecee513dc2e880f56674e7fd157d1dc3a/llama_index/embeddings/text_embeddings_inference.py """

    _user: str = PrivateAttr()
    _timeout: float = PrivateAttr()
    _base_url: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str,
        user: str,
        timeout: float = 60.0,
    ):
        self._user = user
        self._base_url = base_url
        self._timeout = timeout
        super().__init__(
            model_name=model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "EmbeddingEndpoint"

    def _call_api(self, text: str) -> List[float]:
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        return response.json()

    async def _acall_api(self, text: str) -> List[float]:
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        return response.json()

    def _get_query_embedding(self, query: str) -> list[float]:
        """get query embedding."""
        embedding = self._call_api(query)
        return embedding

    def _get_text_embedding(self, text: str) -> list[float]:
        """get text embedding."""
        embedding = self._call_api(text)
        return embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = [self._get_text_embedding(text) for text in texts]
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def embedding_model():
    """Return LlamaIndex embedding model to use for embeddings"""

    from memgpt.config import MemGPTConfig

    # load config
    config = MemGPTConfig.load()

    endpoint = config.embedding_endpoint_type
    if endpoint == "openai":
        model = OpenAIEmbedding(
            api_base=config.embedding_endpoint, api_key=config.openai_key, additional_kwargs={"user": config.anon_clientid}
        )
        return model
    elif endpoint == "azure":
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        model = "text-embedding-ada-002"
        deployment = config.azure_embedding_deployment if config.azure_embedding_deployment is not None else model
        return AzureOpenAIEmbedding(
            model=model,
            deployment_name=deployment,
            api_key=config.azure_key,
            azure_endpoint=config.azure_endpoint,
            api_version=config.azure_version,
        )
    elif endpoint == "hugging-face":
        embed_model = EmbeddingEndpoint(model=config.embedding_model, base_url=config.embedding_endpoint, user=config.anon_clientid)
        return embed_model
    else:
        # default to hugging face model running local
        # warning: this is a terrible model
        from llama_index.embeddings import HuggingFaceEmbedding

        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        model = "BAAI/bge-small-en-v1.5"
        return HuggingFaceEmbedding(model_name=model)
