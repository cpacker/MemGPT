import requests
from typing import Union

from memgpt.models.chat_completion_response import ChatCompletionResponse
from memgpt.models.chat_completion_request import ChatCompletionRequest
from memgpt.models.embedding_response import EmbeddingResponse
from memgpt.utils import smart_urljoin


SUPPORTED_MODELS = [
    "gemini-pro",
]


# TODO use pydantic model as input
def convert_openai_tool_call_to_google_ai(tool_call: dict) -> dict:
    """
    OpenAI format:
    {
      "role": "tool",
      "tool_call_id":
      "content":
    }

    Google AI format:
    {
      "role": "function",
      "parts": [{
        "functionResponse": {
          "name": "find_theaters",
          "response": {
            "name": "find_theaters",
            "content": {
              "movie": "Barbie",
              "theaters": [{
                "name": "AMC Mountain View 16",
                "address": "2000 W El Camino Real, Mountain View, CA 94040"
              }, {
                "name": "Regal Edwards 14",
                "address": "245 Castro St, Mountain View, CA 94040"
              }]
            }
          }
        }]
    }
    """
    pass


# TODO use pydantic model as input
def convert_openai_assistant_content_to_google_ai(tool_call: dict) -> dict:
    pass


# TODO use pydantic model as input
def to_google_ai(openai_message_dict: dict) -> dict:

    # TODO supports "parts" as part of multimodal support
    assert not isinstance(openai_message_dict["content"], list), "Multi-part content is message not yet supported"
    if openai_message_dict["role"] == "user":
        google_ai_message_dict = {
            "role": "user",
            "parts": [{"text": openai_message_dict["content"]}],
        }
    elif openai_message_dict["role"] == "assistant":
        google_ai_message_dict = {
            "role": "model",  # NOTE: diff
            "parts": [{"text": openai_message_dict["content"]}],
        }
    elif openai_message_dict["role"] == "tool":
        google_ai_message_dict = {
            "role": "function",  # NOTE: diff
            "parts": [{"text": openai_message_dict["content"]}],
        }
    else:
        raise ValueError(f"Unsupported conversion (OpenAI -> Google AI) from role {openai_message_dict['role']}")


def google_ai_chat_completions_request(
    service_endpoint: str, model: str, api_key: str, data: ChatCompletionRequest
) -> ChatCompletionResponse:
    """https://ai.google.dev/docs/function_calling

    From https://ai.google.dev/api/rest#service-endpoint:
    "A service endpoint is a base URL that specifies the network address of an API service.
    One service might have multiple service endpoints.
    This service has the following service endpoint and all URIs below are relative to this service endpoint:
    https://xxx.googleapis.com
    """
    from memgpt.utils import printd

    assert service_endpoint is not None, "Missing service_endpoint when calling Google AI"
    assert api_key is not None, "Missing api_key when calling Google AI"
    assert model in SUPPORTED_MODELS, f"Model '{model}' not in supported models: {', '.join(SUPPORTED_MODELS)}"

    url = f"https://{service_endpoint}.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": data["messages"],
    }

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
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
