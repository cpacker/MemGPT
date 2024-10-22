import os
import uuid
from typing import Any, List, Optional

import numpy as np

# from llama_index.core.base.embeddings import BaseEmbedding
# from llama_index.core.embeddings import BaseEmbedding
# from llama_index.core.base.embeddings.base import BaseEmbedding
# from llama_index.bridge.pydantic import PrivateAttr
# from llama_index.embeddings.base import BaseEmbedding
# from llama_index.embeddings.huggingface_utils import format_text
import tiktoken
from llama_index.core import Document as LlamaIndexDocument

# from llama_index.core.base.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

from letta.constants import (
    EMBEDDING_TO_TOKENIZER_DEFAULT,
    EMBEDDING_TO_TOKENIZER_MAP,
    MAX_EMBEDDING_DIM,
)
from letta.schemas.embedding_config import EmbeddingConfig
from letta.utils import is_valid_url, printd


def parse_and_chunk_text(text: str, chunk_size: int) -> List[str]:
    parser = SentenceSplitter(chunk_size=chunk_size)
    llama_index_docs = [LlamaIndexDocument(text=text)]
    nodes = parser.get_nodes_from_documents(llama_index_docs)
    return [n.text for n in nodes]


def truncate_text(text: str, max_length: int, encoding) -> str:
    # truncate the text based on max_length and encoding
    encoded_text = encoding.encode(text)[:max_length]
    return encoding.decode(encoded_text)


def check_and_split_text(text: str, embedding_model: str) -> List[str]:
    """Split text into chunks of max_length tokens or less"""

    if embedding_model in EMBEDDING_TO_TOKENIZER_MAP:
        encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_MAP[embedding_model])
    else:
        print(f"Warning: couldn't find tokenizer for model {embedding_model}, using default tokenizer {EMBEDDING_TO_TOKENIZER_DEFAULT}")
        encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_DEFAULT)

    num_tokens = len(encoding.encode(text))

    # determine max length
    if hasattr(encoding, "max_length"):
        # TODO(fix) this is broken
        max_length = encoding.max_length
    else:
        # TODO: figure out the real number
        printd(f"Warning: couldn't find max_length for tokenizer {embedding_model}, using default max_length 8191")
        max_length = 8191

    # truncate text if too long
    if num_tokens > max_length:
        print(f"Warning: text is too long ({num_tokens} tokens), truncating to {max_length} tokens.")
        # First, apply any necessary formatting
        formatted_text = format_text(text, embedding_model)
        # Then truncate
        text = truncate_text(formatted_text, max_length, encoding)

    return [text]


class EmbeddingEndpoint:
    """Implementation for OpenAI compatible endpoint"""

    # """ Based off llama index https://github.com/run-llama/llama_index/blob/a98bdb8ecee513dc2e880f56674e7fd157d1dc3a/llama_index/embeddings/text_embeddings_inference.py """

    # _user: str = PrivateAttr()
    # _timeout: float = PrivateAttr()
    # _base_url: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str,
        user: str,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        if not is_valid_url(base_url):
            raise ValueError(
                f"Embeddings endpoint was provided an invalid URL (set to: '{base_url}'). Make sure embedding_endpoint is set correctly in your Letta config."
            )
        # TODO: find a neater solution - re-mapping for letta endpoint
        if model == "letta-free":
            model = "BAAI/bge-large-en-v1.5"
        self.model_name = model
        self._user = user
        self._base_url = base_url
        self._timeout = timeout

    def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your Letta config."
            )
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

    def get_text_embedding(self, text: str) -> List[float]:
        return self._call_api(text)


class AzureOpenAIEmbedding:
    def __init__(self, api_endpoint: str, api_key: str, api_version: str, model: str):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=api_endpoint)
        self.model = model

    def get_text_embedding(self, text: str):
        embeddings = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        return embeddings


def default_embedding_model():
    # default to hugging face model running local
    # warning: this is a terrible model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    model = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbedding(model_name=model)


def query_embedding(embedding_model, query_text: str):
    """Generate padded embedding for querying database"""
    query_vec = embedding_model.get_text_embedding(query_text)
    query_vec = np.array(query_vec)
    query_vec = np.pad(query_vec, (0, MAX_EMBEDDING_DIM - query_vec.shape[0]), mode="constant").tolist()
    return query_vec


def embedding_model(config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None):
    """Return LlamaIndex embedding model to use for embeddings"""

    endpoint_type = config.embedding_endpoint_type

    # TODO: refactor to pass in settings from server
    from letta.settings import model_settings

    if endpoint_type == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        additional_kwargs = {"user_id": user_id} if user_id else {}
        model = OpenAIEmbedding(
            api_base=config.embedding_endpoint,
            api_key=model_settings.openai_api_key,
            additional_kwargs=additional_kwargs,
        )
        return model

    elif endpoint_type == "azure":
        assert all(
            [
                model_settings.azure_api_key is not None,
                model_settings.azure_base_url is not None,
                model_settings.azure_api_version is not None,
            ]
        )
        # from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

        ## https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        # model = "text-embedding-ada-002"
        # deployment = credentials.azure_embedding_deployment if credentials.azure_embedding_deployment is not None else model
        # return AzureOpenAIEmbedding(
        #    model=model,
        #    deployment_name=deployment,
        #    api_key=credentials.azure_key,
        #    azure_endpoint=credentials.azure_endpoint,
        #    api_version=credentials.azure_version,
        # )

        return AzureOpenAIEmbedding(
            api_endpoint=model_settings.azure_base_url,
            api_key=model_settings.azure_api_key,
            api_version=model_settings.azure_api_version,
            model=config.embedding_model,
        )

    elif endpoint_type == "hugging-face":
        return EmbeddingEndpoint(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            user=user_id,
        )
    elif endpoint_type == "ollama":

        from llama_index.embeddings.ollama import OllamaEmbedding

        ollama_additional_kwargs = {}
        callback_manager = None

        model = OllamaEmbedding(
            model_name=config.embedding_model,
            base_url=config.embedding_endpoint,
            ollama_additional_kwargs=ollama_additional_kwargs or {},
            callback_manager=callback_manager or None,
        )
        return model

    else:
        return default_embedding_model()
