import typer
import uuid
from typing import Optional, List
import os

from memgpt.utils import is_valid_url
from memgpt.data_types import EmbeddingConfig

from llama_index.embeddings import OpenAIEmbedding, AzureOpenAIEmbedding
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_utils import format_text
import tiktoken


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
        if not is_valid_url(base_url):
            raise ValueError(
                f"Embeddings endpoint was provided an invalid URL (set to: '{base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )
        self._user = user
        self._base_url = base_url
        self._timeout = timeout
        self._encoding = tiktoken.get_encoding(model)
        super().__init__(
            model_name=model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "EmbeddingEndpoint"

    def count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model's tokenizer"""
        return len(self._encoding.encode(text))

    def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )
        import httpx

        # If necessary, truncate text to fit in the embedding model's max sequence length (usually 512)
        num_tokens = self.count_tokens(text)
        max_length = self._encoding.max_length
        if num_tokens > max_length:
            text = format_text(text, self.model_name, max_length=max_length)

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        response_json = response.json()

        if isinstance(response_json, list):
            # embedding directly in response
            embedding = response_json
        elif isinstance(response_json, dict):
            # TEI embedding packaged inside openai-style response
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")
        else:
            # unknown response, can't parse
            raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")

        return embedding

    async def _acall_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )
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
        response_json = response.json()

        if isinstance(response_json, list):
            # embedding directly in response
            embedding = response_json
        elif isinstance(response_json, dict):
            # TEI embedding packaged inside openai-style response
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")
        else:
            # unknown response, can't parse
            raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")

        return embedding

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


def default_embedding_model():
    # default to hugging face model running local
    # warning: this is a terrible model
    from llama_index.embeddings import HuggingFaceEmbedding

    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    model = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbedding(model_name=model)


def embedding_model(config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None):
    """Return LlamaIndex embedding model to use for embeddings"""

    endpoint_type = config.embedding_endpoint_type

    if endpoint_type == "openai":
        additional_kwargs = {"user_id": user_id} if user_id else {}
        model = OpenAIEmbedding(api_base=config.embedding_endpoint, api_key=config.openai_key, additional_kwargs=additional_kwargs)
        return model
    elif endpoint_type == "azure":
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
    elif endpoint_type == "hugging-face":
        try:
            embed_model = EmbeddingEndpoint(model=config.embedding_model, base_url=config.embedding_endpoint, user=user_id)
        except:
            embed_model = default_embedding_model()
        return embed_model
    else:
        return default_embedding_model()
