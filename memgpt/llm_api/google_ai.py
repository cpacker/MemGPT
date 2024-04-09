import requests
from typing import Union, List

from memgpt.models.chat_completion_response import ChatCompletionResponse
from memgpt.models.chat_completion_request import ChatCompletionRequest, Tool
from memgpt.models.embedding_response import EmbeddingResponse
from memgpt.utils import smart_urljoin
from memgpt.constants import NON_USER_MSG_PREFIX

# from memgpt.data_types import ToolCall


SUPPORTED_MODELS = [
    "gemini-pro",
]


def annotate_messages_with_tool_names():
    return


def add_dummy_model_messages(messages: List[dict]) -> List[dict]:
    """Google AI API requires all function call returns are immediately followed by a 'model' role message.

    In MemGPT, the 'model' will often call a function (e.g. send_message) that itself yields to the user,
    so there is no natural follow-up 'model' role message.

    To satisfy the Google AI API restrictions, we can add a dummy 'yield' message
    with role == 'model' that is placed in-betweeen and function output
    (role == 'tool') and user message (role == 'user').
    """
    dummy_yield_message = {"role": "model", "parts": [{"text": f"{NON_USER_MSG_PREFIX}Function call returned, waiting for user response."}]}
    messages_with_padding = []
    for i, message in enumerate(messages):
        messages_with_padding.append(message)
        # Check if the current message role is 'tool' and the next message role is 'user'
        if message["role"] in ["tool", "function"] and (i + 1 < len(messages) and messages[i + 1]["role"] == "user"):
            messages_with_padding.append(dummy_yield_message)

    return messages_with_padding


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


# TODO convert return type to pydantic
def convert_tools_to_google_ai_format(tools: List[Tool]) -> List[dict]:
    """
    OpenAI style:
      "tools": [{
        "type": "function",
        "function": {
            "name": "find_movies",
            "description": "find ....",
            "parameters": {
              ...
            }
        }
      }
      ]

    Google AI style:
      "tools": [{
        "functionDeclarations": [{
          "name": "find_movies",
          "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
          "parameters": {
            "type": "OBJECT",
            "properties": {
              "location": {
                "type": "STRING",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "description": {
                "type": "STRING",
                "description": "Any kind of description including category or genre, title words, attributes, etc."
              }
            },
            "required": ["description"]
          }
        }, {
          "name": "find_theaters",
          ...
    """
    function_list = [
        dict(
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,  # TODO need to unpack
        )
        for t in tools
    ]
    return [{"functionDeclarations": function_list}]


# TODO convert 'data' type to pydantic
def google_ai_chat_completions_request(
    service_endpoint: str,
    model: str,
    api_key: str,
    data: dict,
    key_in_header: bool = True,
    add_postfunc_model_messages: bool = True,
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

    # Two ways to pass the key: https://ai.google.dev/tutorials/setup
    if key_in_header:
        url = f"https://{service_endpoint}.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    else:
        url = f"https://{service_endpoint}.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

    # data["contents"][-1]["role"] = "model"
    if add_postfunc_model_messages:
        data["contents"] = add_dummy_model_messages(data["contents"])

    print(f"messages in 'contents'")
    for m in data["contents"]:
        print(m)

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
        # Print the HTTP status code
        print(f"HTTP Error: {http_err.response.status_code}")
        # Print the response content (error message from server)
        print(f"Message: {http_err.response.text}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e
