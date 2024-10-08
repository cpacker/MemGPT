from typing import List, Optional

from pydantic import BaseModel, Field

from letta.constants import LLM_MAX_TOKENS
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig


class Provider(BaseModel):
    base_url: str

    def list_llm_models(self):
        return []

    def list_embedding_models(self):
        return []

    def get_model_context_window(self, model_name: str):
        pass


class OpenAIProvider(Provider):
    name: str = "openai"
    api_key: str = Field(..., description="API key for the OpenAI API.")
    base_url: str = "https://api.openai.com/v1"

    def list_llm_models(self) -> List[LLMConfig]:
        from letta.llm_api.openai import openai_get_model_list

        response = openai_get_model_list(self.base_url, api_key=self.api_key)
        model_options = [obj["id"] for obj in response["data"]]

        configs = []
        for model_name in model_options:
            context_window_size = self.get_model_context_window_size(model_name)

            if not context_window_size:
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
    base_url: str = Field(..., description="Base URL for the Ollama API.")
    api_key: Optional[str] = Field(None, description="API key for the Ollama API (default: `None`).")

    def list_llm_models(self) -> List[LLMConfig]:
        # https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
        import requests

        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise Exception(f"Failed to list Ollama models: {response.text}")
        response_json = response.json()

        configs = []
        for model in response_json["models"]:
            context_window = self.get_model_context_window(model["name"])
            configs.append(
                LLMConfig(
                    model=model["name"],
                    model_endpoint_type="ollama",
                    model_endpoint=self.base_url,
                    context_window=context_window,
                )
            )
        return configs

    def get_model_context_window(self, model_name: str):

        import requests

        response = requests.post(f"{self.base_url}/api/show", json={"name": model_name, "verbose": True})
        response_json = response.json()

        # thank you vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/config.py#L1675
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # ChatGLM2
            "seq_length",
            # Command-R
            "model_max_length",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]

        # max_position_embeddings
        # parse model cards: nous, dolphon, llama
        for key, value in response_json["model_info"].items():
            if "context_window" in key:
                return value
        return None

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        # TODO: filter embedding models
        return []


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


class GoogleAIProvider(Provider):
    # gemini
    api_key: str = Field(..., description="API key for the Google AI API.")
    service_endpoint: str = "generativelanguage"
    base_url: str = "https://generativelanguage.googleapis.com"

    def list_llm_models(self):
        from letta.llm_api.google_ai import google_ai_get_model_list

        # TODO: use base_url instead
        model_options = google_ai_get_model_list(service_endpoint=self.service_endpoint, api_key=self.api_key)
        model_options = [str(m["name"]) for m in model_options]
        model_options = [mo[len("models/") :] if mo.startswith("models/") else mo for mo in model_options]
        # TODO remove manual filtering for gemini-pro
        model_options = [mo for mo in model_options if str(mo).startswith("gemini") and "-pro" in str(mo)]
        # TODO: add context windows
        # model_options = ["gemini-pro"]

        configs = []
        for model in model_options:
            configs.append(
                LLMConfig(
                    model=model,
                    model_endpoint_type="google_ai",
                    model_endpoint=self.base_url,
                    context_window=self.get_model_context_window(model),
                )
            )
        return configs

    def list_embedding_models(self):
        return []

    def get_model_context_window(self, model_name: str):
        from letta.llm_api.google_ai import google_ai_get_model_context_window

        # TODO: use base_url instead
        return google_ai_get_model_context_window(self.service_endpoint, self.api_key, model_name)


class AzureProvider(Provider):
    name: str = "azure"
    base_url: str = Field(
        ..., description="Base URL for the Azure API endpoint. This should be specific to your org, e.g. `https://letta.openai.azure.com`."
    )
    api_key: str = Field(..., description="API key for the Azure API.")


class VLLMProvider(OpenAIProvider):
    # NOTE: vLLM only serves one model at a time (so could configure that through env variables)
    pass


class CohereProvider(OpenAIProvider):
    pass
