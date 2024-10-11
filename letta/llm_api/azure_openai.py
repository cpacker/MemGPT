import requests

from letta.llm_api.helpers import make_post_request
from letta.schemas.llm_config import LLMConfig
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse
from letta.schemas.openai.chat_completions import ChatCompletionRequest
from letta.schemas.openai.embedding_response import EmbeddingResponse
from letta.settings import ModelSettings


def get_azure_chat_completions_endpoint(base_url: str, model: str, api_version: str):
    return f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"


def get_azure_embeddings_endpoint(base_url: str, model: str, api_version: str):
    return f"{base_url}/openai/deployments/{model}/embeddings?api-version={api_version}"


def get_azure_model_list_endpoint(base_url: str, api_version: str):
    return f"{base_url}/openai/models?api-version={api_version}"


def azure_openai_get_model_list(base_url: str, api_key: str, api_version: str) -> list:
    """https://learn.microsoft.com/en-us/rest/api/azureopenai/models/list?view=rest-azureopenai-2023-05-15&tabs=HTTP"""

    # https://xxx.openai.azure.com/openai/models?api-version=xxx
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["api-key"] = f"{api_key}"

    url = get_azure_model_list_endpoint(base_url, api_version)
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to retrieve model list: {e}")

    return response.json().get("data", [])


def azure_openai_get_chat_completion_model_list(base_url: str, api_key: str, api_version: str) -> list:
    model_list = azure_openai_get_model_list(base_url, api_key, api_version)
    # Extract models that support text generation
    model_options = [m for m in model_list if m.get("capabilities").get("chat_completion") == True]
    return model_options


def azure_openai_get_embeddings_model_list(base_url: str, api_key: str, api_version: str, require_embedding_in_name: bool = True) -> list:
    def valid_embedding_model(m: dict):
        valid_name = True
        if require_embedding_in_name:
            valid_name = "embedding" in m["id"]

        return m.get("capabilities").get("embeddings") == True and valid_name

    model_list = azure_openai_get_model_list(base_url, api_key, api_version)
    # Extract models that support embeddings

    model_options = [m for m in model_list if valid_embedding_model(m)]
    return model_options


def azure_openai_chat_completions_request(
    model_settings: ModelSettings, llm_config: LLMConfig, api_key: str, chat_completion_request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions"""

    assert api_key is not None, "Missing required field when calling Azure OpenAI"

    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}
    data = chat_completion_request.model_dump(exclude_none=True)

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")

    url = get_azure_chat_completions_endpoint(model_settings.azure_base_url, llm_config.model, model_settings.azure_api_version)
    response_json = make_post_request(url, headers, data)
    # NOTE: azure openai does not include "content" in the response when it is None, so we need to add it
    if "content" not in response_json["choices"][0].get("message"):
        response_json["choices"][0]["message"]["content"] = None
    response = ChatCompletionResponse(**response_json)  # convert to 'dot-dict' style which is the openai python client default
    return response


def azure_openai_embeddings_request(
    resource_name: str, deployment_id: str, api_version: str, api_key: str, data: dict
) -> EmbeddingResponse:
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings"""

    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}

    response_json = make_post_request(url, headers, data)
    return EmbeddingResponse(**response_json)
