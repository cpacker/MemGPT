import requests
import uuid
import time
from typing import Union, Optional, List

from memgpt.data_types import Message
from memgpt.models.chat_completion_response import ChatCompletionResponse
from memgpt.models.chat_completion_request import ChatCompletionRequest, Tool
from memgpt.models.embedding_response import EmbeddingResponse
from memgpt.utils import smart_urljoin


def openai_get_model_context_window(url: str, api_key: Union[str, None], model: str, fix_url: Optional[bool] = False) -> str:
    # NOTE: this actually doesn't work for OpenAI atm, just some OpenAI-compatible APIs like Groq
    model_list = openai_get_model_list(url=url, api_key=api_key, fix_url=fix_url)

    for model_dict in model_list["data"]:
        if model_dict["id"] == model and "context_window" in model_dict:
            return int(model_dict["context_window"])
    raise ValueError(f"Can't find model '{model}' in model list")


def openai_get_model_list(url: str, api_key: Union[str, None], fix_url: Optional[bool] = False) -> dict:
    """https://platform.openai.com/docs/api-reference/models/list"""
    from memgpt.utils import printd

    # In some cases we may want to double-check the URL and do basic correction, eg:
    # In MemGPT config the address for vLLM is w/o a /v1 suffix for simplicity
    # However if we're treating the server as an OpenAI proxy we want the /v1 suffix on our model hit
    if fix_url:
        if not url.endswith("/v1"):
            url = smart_urljoin(url, "v1")

    url = smart_urljoin(url, "models")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

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


def add_inner_thoughts_to_tool_params(tools: List[Tool], inner_thoughts_required: Optional[bool] = True) -> List[Tool]:
    from memgpt.local_llm.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION

    tools_with_inner_thoughts = []
    for tool in tools:
        assert INNER_THOUGHTS_KWARG not in tool.function.parameters["properties"], tool

        tool.function.parameters["properties"][INNER_THOUGHTS_KWARG] = {
            "type": "string",
            "description": INNER_THOUGHTS_KWARG_DESCRIPTION,
        }

        if inner_thoughts_required:
            assert INNER_THOUGHTS_KWARG not in tool.function.parameters["required"], tool
            tool.function.parameters["required"].append(INNER_THOUGHTS_KWARG)

        tools_with_inner_thoughts.append(tool)

    return tools_with_inner_thoughts


def openai_chat_completions_request(
    url: str, api_key: str, chat_completion_request: ChatCompletionRequest, inner_thoughts_in_tools: Optional[bool] = False
) -> ChatCompletionResponse:
    """https://platform.openai.com/docs/guides/text-generation?lang=curl"""
    from memgpt.utils import printd

    url = smart_urljoin(url, "chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Certain inference backends or models may not support non-null content + function call
    # E.g., Groq's API is technically OpenAI compliant, but does not tend to (always?) returns non-null contents when function calling
    # In this case, we may want to move inner thoughts into the function parameters to ensure that we still get CoT
    if inner_thoughts_in_tools:
        chat_completion_request.tools = add_inner_thoughts_to_tool_params(chat_completion_request.tools)

    data = chat_completion_request.model_dump(exclude_none=True)

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")

    print("aaa")
    for m in data["messages"]:
        print(m)

    if inner_thoughts_in_tools:
        # move inner thoughts to func calls in the chat history by recasting
        msg_objs = [
            Message.dict_to_message(user_id=uuid.uuid4(), agent_id=uuid.uuid4(), openai_message_dict=m, inner_thoughts_in_kwargs=True)
            for m in data["messages"]
        ]
        data["messages"] = [m.to_openai_dict(put_inner_thoughts_in_kwargs=True) for m in msg_objs]

        print("zzz")
        for m in data["messages"]:
            print(m)

        print("xxx")
        for t in data["tools"]:
            print(t)

    printd(f"Sending request to {url}")
    try:
        # Example code to trigger a rate limit response:
        # mock_response = requests.Response()
        # mock_response.status_code = 429
        # http_error = requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
        # http_error.response = mock_response
        # raise http_error

        # Example code to trigger a context overflow response (for an 8k model)
        # data["messages"][-1]["content"] = " ".join(["repeat after me this is not a fluke"] * 1000)

        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        response = ChatCompletionResponse(**response)  # convert to 'dot-dict' style which is the openai python client default

        if inner_thoughts_in_tools:
            # We need to strip the inner thought out of the parameters and put it back inside the content
            raise NotImplementedError

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


def openai_embeddings_request(url: str, api_key: str, data: dict) -> EmbeddingResponse:
    """https://platform.openai.com/docs/api-reference/embeddings/create"""
    from memgpt.utils import printd

    url = smart_urljoin(url, "embeddings")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

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
