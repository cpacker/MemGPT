from typing import List, Optional

from pydantic import BaseModel, Field

from letta.constants import LLM_MAX_TOKENS
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig


class Provider(BaseModel):
    base_url: str

    def list_llm_models(self):
        pass

    def list_embedding_models(self):
        pass

    def get_model_context_window(self, model_name: str):
        pass


class OpenAIProvider(Provider):
    name: str = "openai"
    api_key: str = Field(..., description="API key for the OpenAI API.")
    base_url: str = "https://api.openai.com/v1"

    def list_llm_models(self) -> List[LLMConfig]:
        from letta.llm_api.openai import openai_get_model_list

        response = openai_get_model_list(self.base_url, api_key=self.api_key)
        print(response)
        print(response["data"])
        model_options = [obj["id"] for obj in response["data"]]

        configs = []
        for model_name in model_options:
            print(model_name)
            context_window_size = self.get_model_context_window_size(model_name)

            if not context_window_size:
                print(f"Missing context window information for {model_name}, skipping")
                continue
            configs.append(
                LLMConfig(model=model_name, model_endpoint_type="openai", model_endpoint=self.base_url, context_window=context_window_size)
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:

        # TODO: actually automatically list models
        return [
            EmbeddingConfig(
                embedding_model="text-embedding-ada-002",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=300,
            )
        ]

    def get_model_context_window_size(self, model_name: str):
        if model_name in LLM_MAX_TOKENS:
            return LLM_MAX_TOKENS[model_name]
        else:
            return None


class AnthropicProvider(Provider):
    name: str = "anthropic"
    api_key: str = Field(..., description="API key for the Anthropic API.")
    base_url: str = "https://api.anthropic.com/v1"

    def list_llm_models(self) -> List[LLMConfig]:
        from letta.llm_api.anthropic import anthropic_get_model_list

        models = anthropic_get_model_list(self.base_url, api_key=self.api_key)

        configs = []
        for model in models:
            configs.append(
                LLMConfig(
                    model=model["name"],
                    model_endpoint_type="anthropic",
                    model_endpoint=self.base_url,
                    context_window=model["context_window"],
                )
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        return []


class OllamaProvider(OpenAIProvider):
    name: str = "ollama"
    base_url: str = Field("http://localhost:11434", description="Base URL for the Ollama API.")
    api_key: Optional[str] = Field(None, description="API key for the Ollama API (default: `None`).")

    def list_llm_models(self) -> List[LLMConfig]:
        # https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
        import requests

        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise Exception(f"Failed to list Ollama models: {response.text}")
        response_json = response.json()
        print(response_json)

        configs = []
        for model in response_json["models"]:
            configs.append(
                LLMConfig(
                    model=model["name"], model_endpoint_type="ollama", model_endpoint=self.base_url, context_window=model["context_window"]
                )
            )
        return configs


class GroqProvider(OpenAIProvider):
    name: str = "groq"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = Field(..., description="API key for the Groq API.")

    def list_llm_models(self) -> List[LLMConfig]:
        from letta.llm_api.openai import openai_get_model_list

        response = openai_get_model_list(self.base_url, api_key=self.api_key)
        configs = []
        for model in response["data"]:
            if not "context_window" in model:
                print(f"Missing context window information for {model['id']}, skipping")
                continue
            configs.append(
                LLMConfig(
                    model=model["id"], model_endpoint_type="openai", model_endpoint=self.base_url, context_window=model["context_window"]
                )
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        return []

    def get_model_context_window_size(self, model_name: str):
        raise NotImplementedError
