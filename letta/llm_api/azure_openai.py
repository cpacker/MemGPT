from typing import Union

import requests

from letta.schemas.llm_config import LLMConfig
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse
from letta.schemas.openai.chat_completions import ChatCompletionRequest
from letta.schemas.openai.embedding_response import EmbeddingResponse
from letta.settings import ModelSettings
from letta.utils import smart_urljoin

MODEL_TO_AZURE_ENGINE = {
    "gpt-4-1106-preview": "gpt-4",
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4-32k",
    "gpt-3.5": "gpt-35-turbo",
    "gpt-3.5-turbo": "gpt-35-turbo",
    "gpt-3.5-turbo-16k": "gpt-35-turbo-16k",
    "gpt-4o-mini": "gpt-4o-mini",
}


def get_azure_endpoint(llm_config: LLMConfig, model_settings: ModelSettings):
    assert llm_config.api_version, "Missing model version! This field must be provided in the LLM config for Azure."
    assert llm_config.model in MODEL_TO_AZURE_ENGINE, f"{llm_config.model} not in supported models: {list(MODEL_TO_AZURE_ENGINE.keys())}"

    model = MODEL_TO_AZURE_ENGINE[llm_config.model]
    return f"{model_settings.azure_base_url}/openai/deployments/{model}/chat/completions?api-version={llm_config.api_version}"


def azure_openai_get_model_list(url: str, api_key: Union[str, None], api_version: str) -> dict:
    """https://learn.microsoft.com/en-us/rest/api/azureopenai/models/list?view=rest-azureopenai-2023-05-15&tabs=HTTP"""
    from letta.utils import printd

    # https://xxx.openai.azure.com/openai/models?api-version=xxx
    url = smart_urljoin(url, "openai")
    url = smart_urljoin(url, f"models?api-version={api_version}")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["api-key"] = f"{api_key}"

    printd(f"Sending request to {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response = {response}")
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        try:
            response = response.json()
        except:
            pass
        printd(f"Got HTTPError, exception={http_err}, response={response}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        try:
            response = response.json()
        except:
            pass
        printd(f"Got RequestException, exception={req_err}, response={response}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        try:
            response = response.json()
        except:
            pass
        printd(f"Got unknown Exception, exception={e}, response={response}")
        raise e


def azure_openai_chat_completions_request(
    model_settings: ModelSettings, llm_config: LLMConfig, api_key: str, chat_completion_request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions"""
    from letta.utils import printd

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

    model_endpoint = get_azure_endpoint(llm_config, model_settings)
    printd(f"Sending request to {model_endpoint}")
    try:
        response = requests.post(model_endpoint, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        # NOTE: azure openai does not include "content" in the response when it is None, so we need to add it
        if "content" not in response["choices"][0].get("message"):
            response["choices"][0]["message"]["content"] = None
        response = ChatCompletionResponse(**response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}, payload={data}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def azure_openai_embeddings_request(
    resource_name: str, deployment_id: str, api_version: str, api_key: str, data: dict
) -> EmbeddingResponse:
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings"""
    from letta.utils import printd

    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        response = EmbeddingResponse(**response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}, payload={data}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e
