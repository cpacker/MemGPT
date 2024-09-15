import json
import uuid
from typing import List, Optional, Union

import requests

from letta.local_llm.utils import count_tokens
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_request import ChatCompletionRequest, Tool
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
)
from letta.schemas.openai.chat_completion_response import (
    Message as ChoiceMessage,  # NOTE: avoid conflict with our own Letta Message datatype
)
from letta.schemas.openai.chat_completion_response import ToolCall, UsageStatistics
from letta.utils import get_tool_call_id, get_utc_time, json_dumps, smart_urljoin

BASE_URL = "https://api.cohere.ai/v1"

# models that we know will work with Letta
COHERE_VALID_MODEL_LIST = [
    "command-r-plus",
]


def cohere_get_model_details(url: str, api_key: Union[str, None], model: str) -> int:
    """https://docs.cohere.com/reference/get-model"""
    from letta.utils import printd

    url = smart_urljoin(url, "models")
    url = smart_urljoin(url, model)
    headers = {
        "accept": "application/json",
        "authorization": f"bearer {api_key}",
    }

    printd(f"Sending request to {url}")
    try:
        response = requests.get(url, headers=headers)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def cohere_get_model_context_window(url: str, api_key: Union[str, None], model: str) -> int:
    model_details = cohere_get_model_details(url=url, api_key=api_key, model=model)
    return model_details["context_length"]


def cohere_get_model_list(url: str, api_key: Union[str, None]) -> dict:
    """https://docs.cohere.com/reference/list-models"""
    from letta.utils import printd

    url = smart_urljoin(url, "models")
    headers = {
        "accept": "application/json",
        "authorization": f"bearer {api_key}",
    }

    printd(f"Sending request to {url}")
    try:
        response = requests.get(url, headers=headers)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        return response["models"]
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def remap_finish_reason(finish_reason: str) -> str:
    """Remap Cohere's 'finish_reason' to OpenAI 'finish_reason'

    OpenAI: 'stop', 'length', 'function_call', 'content_filter', null
    see: https://platform.openai.com/docs/guides/text-generation/chat-completions-api

    Cohere finish_reason is different but undocumented ???
    """
    if finish_reason == "COMPLETE":
        return "stop"
    elif finish_reason == "MAX_TOKENS":
        return "length"
    # elif stop_reason == "tool_use":
    # return "function_call"
    else:
        raise ValueError(f"Unexpected stop_reason: {finish_reason}")


def convert_cohere_response_to_chatcompletion(
    response_json: dict,  # REST response from API
    model: str,  # Required since not returned
    inner_thoughts_in_kwargs: Optional[bool] = True,
) -> ChatCompletionResponse:
    """
    Example response from command-r-plus:
    response.json = {
        'response_id': '28c47751-acce-41cd-8c89-c48a15ac33cf',
        'text': '',
        'generation_id': '84209c9e-2868-4984-82c5-063b748b7776',
        'chat_history': [
            {
                'role': 'CHATBOT',
                'message': 'Bootup sequence complete. Persona activated. Testing messaging functionality.'
            },
            {
                'role': 'SYSTEM',
                'message': '{"status": "OK", "message": null, "time": "2024-04-11 11:22:36 PM PDT-0700"}'
            }
        ],
        'finish_reason': 'COMPLETE',
        'meta': {
            'api_version': {'version': '1'},
            'billed_units': {'input_tokens': 692, 'output_tokens': 20},
            'tokens': {'output_tokens': 20}
        },
        'tool_calls': [
            {
                'name': 'send_message',
                'parameters': {
                    'message': "Hello Chad, it's Sam. How are you feeling today?"
                }
            }
        ]
    }
    """
    if "billed_units" in response_json["meta"]:
        prompt_tokens = response_json["meta"]["billed_units"]["input_tokens"]
        completion_tokens = response_json["meta"]["billed_units"]["output_tokens"]
    else:
        # For some reason input_tokens not included in 'meta' 'tokens' dict?
        prompt_tokens = count_tokens(json_dumps(response_json["chat_history"]))  # NOTE: this is a very rough approximation
        completion_tokens = response_json["meta"]["tokens"]["output_tokens"]

    finish_reason = remap_finish_reason(response_json["finish_reason"])

    if "tool_calls" in response_json and response_json["tool_calls"] is not None:
        inner_thoughts = []
        tool_calls = []
        for tool_call_response in response_json["tool_calls"]:
            function_name = tool_call_response["name"]
            function_args = tool_call_response["parameters"]
            if inner_thoughts_in_kwargs:
                from letta.local_llm.constants import INNER_THOUGHTS_KWARG

                assert INNER_THOUGHTS_KWARG in function_args
                # NOTE:
                inner_thoughts.append(function_args.pop(INNER_THOUGHTS_KWARG))

            tool_calls.append(
                ToolCall(
                    id=get_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=function_name,
                        arguments=json.dumps(function_args),
                    ),
                )
            )

        # NOTE: no multi-call support for now
        assert len(tool_calls) == 1, tool_calls
        content = inner_thoughts[0]

    else:
        # raise NotImplementedError(f"Expected a tool call response from Cohere API")
        content = response_json["text"]
        tool_calls = None

    # In Cohere API empty string == null
    content = None if content == "" else content
    assert content is not None or tool_calls is not None, "Response message must have either content or tool_calls"

    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        message=ChoiceMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
    )

    return ChatCompletionResponse(
        id=response_json["response_id"],
        choices=[choice],
        created=get_utc_time(),
        model=model,
        usage=UsageStatistics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def convert_tools_to_cohere_format(tools: List[Tool], inner_thoughts_in_kwargs: Optional[bool] = True) -> List[dict]:
    """See: https://docs.cohere.com/reference/chat

    OpenAI style:
      "tools": [{
        "type": "function",
        "function": {
            "name": "find_movies",
            "description": "find ....",
            "parameters": {
              "type": "object",
              "properties": {
                 PARAM: {
                   "type": PARAM_TYPE,  # eg "string"
                   "description": PARAM_DESCRIPTION,
                 },
                 ...
              },
              "required": List[str],
            }
        }
      }]

    Cohere style:
      "tools": [{
        "name": "find_movies",
        "description": "find ....",
        "parameter_definitions": {
            PARAM_NAME: {
                "description": PARAM_DESCRIPTION,
                "type": PARAM_TYPE,  # eg "string"
                "required": <boolean>,
            }
          },
        }
      }]
    """
    tools_dict_list = []
    for tool in tools:
        tools_dict_list.append(
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameter_definitions": {
                    p_name: {
                        "description": p_fields["description"],
                        "type": p_fields["type"],
                        "required": p_name in tool.function.parameters["required"],
                    }
                    for p_name, p_fields in tool.function.parameters["properties"].items()
                },
            }
        )

    if inner_thoughts_in_kwargs:
        # NOTE: since Cohere doesn't allow "text" in the response when a tool call happens, if we want
        # a simultaneous CoT + tool call we need to put it inside a kwarg
        from letta.local_llm.constants import (
            INNER_THOUGHTS_KWARG,
            INNER_THOUGHTS_KWARG_DESCRIPTION,
        )

        for cohere_tool in tools_dict_list:
            cohere_tool["parameter_definitions"][INNER_THOUGHTS_KWARG] = {
                "description": INNER_THOUGHTS_KWARG_DESCRIPTION,
                "type": "string",
                "required": True,
            }

    return tools_dict_list


def cohere_chat_completions_request(
    url: str,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """https://docs.cohere.com/docs/multi-step-tool-use"""
    from letta.utils import printd

    url = smart_urljoin(url, "chat")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"bearer {api_key}",
    }

    # convert the tools
    cohere_tools = None if chat_completion_request.tools is None else convert_tools_to_cohere_format(chat_completion_request.tools)

    # pydantic -> dict
    data = chat_completion_request.model_dump(exclude_none=True)

    if "functions" in data:
        raise ValueError(f"'functions' unexpected in Anthropic API payload")

    # If tools == None, strip from the payload
    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")

    # Convert messages to Cohere format
    msg_objs = [Message.dict_to_message(user_id=uuid.uuid4(), agent_id=uuid.uuid4(), openai_message_dict=m) for m in data["messages"]]

    # System message 0 should instead be a "preamble"
    # See: https://docs.cohere.com/reference/chat
    # The chat_history parameter should not be used for SYSTEM messages in most cases. Instead, to add a SYSTEM role message at the beginning of a conversation, the preamble parameter should be used.
    assert msg_objs[0].role == "system", msg_objs[0]
    preamble = msg_objs[0].text

    # data["messages"] = [m.to_cohere_dict() for m in msg_objs[1:]]
    data["messages"] = []
    for m in msg_objs[1:]:
        ms = m.to_cohere_dict()  # NOTE: returns List[dict]
        data["messages"].extend(ms)

    assert data["messages"][-1]["role"] == "USER", data["messages"][-1]
    data = {
        "preamble": preamble,
        "chat_history": data["messages"][:-1],
        "message": data["messages"][-1]["message"],
        "tools": cohere_tools,
    }

    # Move 'system' to the top level
    # 'messages: Unexpected role "system". The Messages API accepts a top-level `system` parameter, not "system" as an input message role.'
    # assert data["messages"][0]["role"] == "system", f"Expected 'system' role in messages[0]:\n{data['messages'][0]}"
    # data["system"] = data["messages"][0]["content"]
    # data["messages"] = data["messages"][1:]

    # Convert to Anthropic format
    # msg_objs = [Message.dict_to_message(user_id=uuid.uuid4(), agent_id=uuid.uuid4(), openai_message_dict=m) for m in data["messages"]]
    # data["messages"] = [m.to_anthropic_dict(inner_thoughts_xml_tag=inner_thoughts_xml_tag) for m in msg_objs]

    # Handling Anthropic special requirement for 'user' message in front
    # messages: first message must use the "user" role'
    # if data["messages"][0]["role"] != "user":
    # data["messages"] = [{"role": "user", "content": DUMMY_FIRST_USER_MESSAGE}] + data["messages"]

    # Handle Anthropic's restriction on alternating user/assistant messages
    # data["messages"] = merge_tool_results_into_user_messages(data["messages"])

    # Anthropic also wants max_tokens in the input
    # It's also part of ChatCompletions
    # assert "max_tokens" in data, data

    # Remove extra fields used by OpenAI but not Anthropic
    # data.pop("frequency_penalty", None)
    # data.pop("logprobs", None)
    # data.pop("n", None)
    # data.pop("top_p", None)
    # data.pop("presence_penalty", None)
    # data.pop("user", None)
    # data.pop("tool_choice", None)

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        response = convert_cohere_response_to_chatcompletion(response_json=response, model=chat_completion_request.model)
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
