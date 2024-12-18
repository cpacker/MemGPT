import json
import re
from typing import List, Optional, Union

from letta.llm_api.helpers import make_post_request
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
from letta.utils import get_utc_time, smart_urljoin

BASE_URL = "https://api.anthropic.com/v1"


# https://docs.anthropic.com/claude/docs/models-overview
# Sadly hardcoded
MODEL_LIST = [
    {
        "name": "claude-3-opus-20240229",
        "context_window": 200000,
    },
    {
        "name": "claude-3-5-sonnet-20241022",
        "context_window": 200000,
    },
    {
        "name": "claude-3-5-haiku-20241022",
        "context_window": 200000,
    },
]

DUMMY_FIRST_USER_MESSAGE = "User initializing bootup sequence."


def antropic_get_model_context_window(url: str, api_key: Union[str, None], model: str) -> int:
    for model_dict in anthropic_get_model_list(url=url, api_key=api_key):
        if model_dict["name"] == model:
            return model_dict["context_window"]
    raise ValueError(f"Can't find model '{model}' in Anthropic model list")


def anthropic_get_model_list(url: str, api_key: Union[str, None]) -> dict:
    """https://docs.anthropic.com/claude/docs/models-overview"""

    # NOTE: currently there is no GET /models, so we need to hardcode
    return MODEL_LIST


def convert_tools_to_anthropic_format(tools: List[Tool]) -> List[dict]:
    """See: https://docs.anthropic.com/claude/docs/tool-use

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
      }
      ]

    Anthropic style:
      "tools": [{
        "name": "find_movies",
        "description": "find ....",
        "input_schema": {
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
      ]

      Two small differences:
        - 1 level less of nesting
        - "parameters" -> "input_schema"
    """
    formatted_tools = []
    for tool in tools:
        formatted_tool = {
            "name"         : tool.function.name,
            "description"  : tool.function.description,
            "input_schema"   : tool.function.parameters or {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        formatted_tools.append(formatted_tool)

    return formatted_tools


def merge_tool_results_into_user_messages(messages: List[dict]):
    """Anthropic API doesn't allow role 'tool'->'user' sequences

    Example HTTP error:
    messages: roles must alternate between "user" and "assistant", but found multiple "user" roles in a row

    From: https://docs.anthropic.com/claude/docs/tool-use
    You may be familiar with other APIs that return tool use as separate from the model's primary output,
    or which use a special-purpose tool or function message role.
    In contrast, Anthropic's models and API are built around alternating user and assistant messages,
    where each message is an array of rich content blocks: text, image, tool_use, and tool_result.
    """

    # TODO walk through the messages list
    # When a dict (dict_A) with 'role' == 'user' is followed by a dict with 'role' == 'user' (dict B), do the following
    # dict_A["content"] = dict_A["content"] + dict_B["content"]

    # The result should be a new merged_messages list that doesn't have any back-to-back dicts with 'role' == 'user'
    merged_messages = []
    if not messages:
        return merged_messages

    # Start with the first message in the list
    current_message = messages[0]

    for next_message in messages[1:]:
        if current_message["role"] == "user" and next_message["role"] == "user":
            # Merge contents of the next user message into current one
            current_content = (
                current_message["content"]
                if isinstance(current_message["content"], list)
                else [{"type": "text", "text": current_message["content"]}]
            )
            next_content = (
                next_message["content"]
                if isinstance(next_message["content"], list)
                else [{"type": "text", "text": next_message["content"]}]
            )
            merged_content = current_content + next_content
            current_message["content"] = merged_content
        else:
            # Append the current message to result as it's complete
            merged_messages.append(current_message)
            # Move on to the next message
            current_message = next_message

    # Append the last processed message to the result
    merged_messages.append(current_message)

    return merged_messages


def remap_finish_reason(stop_reason: str) -> str:
    """Remap Anthropic's 'stop_reason' to OpenAI 'finish_reason'

    OpenAI: 'stop', 'length', 'function_call', 'content_filter', null
    see: https://platform.openai.com/docs/guides/text-generation/chat-completions-api

    From: https://docs.anthropic.com/claude/reference/migrating-from-text-completions-to-messages#stop-reason

    Messages have a stop_reason of one of the following values:
        "end_turn": The conversational turn ended naturally.
        "stop_sequence": One of your specified custom stop sequences was generated.
        "max_tokens": (unchanged)

    """
    if stop_reason == "end_turn":
        return "stop"
    elif stop_reason == "stop_sequence":
        return "stop"
    elif stop_reason == "max_tokens":
        return "length"
    elif stop_reason == "tool_use":
        return "function_call"
    else:
        raise ValueError(f"Unexpected stop_reason: {stop_reason}")


def strip_xml_tags(string: str, tag: Optional[str]) -> str:
    if tag is None:
        return string
    # Construct the regular expression pattern to find the start and end tags
    tag_pattern = f"<{tag}.*?>|</{tag}>"
    # Use the regular expression to replace the tags with an empty string
    return re.sub(tag_pattern, "", string)


def convert_anthropic_response_to_chatcompletion(
    response_json: dict,  # REST response from Google AI API
    inner_thoughts_xml_tag: Optional[str] = None,
) -> ChatCompletionResponse:
    """
    Example response from Claude 3:
    response.json = {
        'id': 'msg_01W1xg9hdRzbeN2CfZM7zD2w',
        'type': 'message',
        'role': 'assistant',
        'content': [
            {
                'type': 'text',
                'text': "<thinking>Analyzing user login event. This is Chad's first
    interaction with me. I will adjust my personality and rapport accordingly.</thinking>"
            },
            {
                'type':
                'tool_use',
                'id': 'toolu_01Ka4AuCmfvxiidnBZuNfP1u',
                'name': 'core_memory_append',
                'input': {
                    'name': 'human',
                    'content': 'Chad is logging in for the first time. I will aim to build a warm
    and welcoming rapport.',
                    'request_heartbeat': True
                }
            }
        ],
        'model': 'claude-3-haiku-20240307',
        'stop_reason': 'tool_use',
        'stop_sequence': None,
        'usage': {
            'input_tokens': 3305,
            'output_tokens': 141
        }
    }
    """
    prompt_tokens = response_json["usage"]["input_tokens"]
    completion_tokens = response_json["usage"]["output_tokens"]

    finish_reason = remap_finish_reason(response_json["stop_reason"])

    if isinstance(response_json["content"], list):
        if len(response_json["content"]) > 1:
            # inner mono + function call
            assert len(response_json["content"]) == 2, response_json
            assert response_json["content"][0]["type"] == "text", response_json
            assert response_json["content"][1]["type"] == "tool_use", response_json
            content = strip_xml_tags(string=response_json["content"][0]["text"], tag=inner_thoughts_xml_tag)
            tool_calls = [
                ToolCall(
                    id=response_json["content"][1]["id"],
                    type="function",
                    function=FunctionCall(
                        name=response_json["content"][1]["name"],
                        arguments=json.dumps(response_json["content"][1]["input"], indent=2),
                    ),
                )
            ]
        elif len(response_json["content"]) == 1:
            if response_json["content"][0]["type"] == "tool_use":
                # function call only
                content = None
                tool_calls = [
                    ToolCall(
                        id=response_json["content"][0]["id"],
                        type="function",
                        function=FunctionCall(
                            name=response_json["content"][0]["name"],
                            arguments=json.dumps(response_json["content"][0]["input"], indent=2),
                        ),
                    )
                ]
            else:
                # inner mono only
                content = strip_xml_tags(string=response_json["content"][0]["text"], tag=inner_thoughts_xml_tag)
                tool_calls = None
    else:
        raise RuntimeError("Unexpected type for content in response_json.")

    assert response_json["role"] == "assistant", response_json
    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        message=ChoiceMessage(
            role=response_json["role"],
            content=content,
            tool_calls=tool_calls,
        ),
    )

    return ChatCompletionResponse(
        id=response_json["id"],
        choices=[choice],
        created=get_utc_time(),
        model=response_json["model"],
        usage=UsageStatistics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def anthropic_chat_completions_request(
    url: str,
    api_key: str,
    data: ChatCompletionRequest,
    inner_thoughts_xml_tag: Optional[str] = "thinking",
) -> ChatCompletionResponse:
    """https://docs.anthropic.com/claude/docs/tool-use"""

    url = smart_urljoin(url, "messages")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        # NOTE: beta headers for tool calling
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024-04-04",
    }

    # convert the tools
    anthropic_tools = None if data.tools is None else convert_tools_to_anthropic_format(data.tools)

    # pydantic -> dict
    data = data.model_dump(exclude_none=True)

    if "functions" in data:
        raise ValueError(f"'functions' unexpected in Anthropic API payload")

    # If tools == None, strip from the payload
    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")
    # Remap to our converted tools
    if anthropic_tools is not None:
        data["tools"] = anthropic_tools

        # TODO: Add support for other tool_choice options like "auto", "any"
        if len(anthropic_tools) == 1:
            data["tool_choice"] = {
                "type": "tool",  # Changed from "function" to "tool"
                "name": anthropic_tools[0]["name"],  # Directly specify name without nested "function" object
                "disable_parallel_tool_use": True  # Force single tool use
            }

    # Move 'system' to the top level
    # 'messages: Unexpected role "system". The Messages API accepts a top-level `system` parameter, not "system" as an input message role.'
    assert data["messages"][0]["role"] == "system", f"Expected 'system' role in messages[0]:\n{data['messages'][0]}"
    data["system"] = data["messages"][0]["content"]
    data["messages"] = data["messages"][1:]

    # set `content` to None if missing
    for message in data["messages"]:
        if "content" not in message:
            message["content"] = None

    # Convert to Anthropic format

    msg_objs = [Message.dict_to_message(user_id=None, agent_id=None, openai_message_dict=m) for m in data["messages"]]
    data["messages"] = [m.to_anthropic_dict(inner_thoughts_xml_tag=inner_thoughts_xml_tag) for m in msg_objs]

    # Handling Anthropic special requirement for 'user' message in front
    # messages: first message must use the "user" role'
    if data["messages"][0]["role"] != "user":
        data["messages"] = [{"role": "user", "content": DUMMY_FIRST_USER_MESSAGE}] + data["messages"]

    # Handle Anthropic's restriction on alternating user/assistant messages
    data["messages"] = merge_tool_results_into_user_messages(data["messages"])

    # Anthropic also wants max_tokens in the input
    # It's also part of ChatCompletions
    assert "max_tokens" in data, data

    # Remove extra fields used by OpenAI but not Anthropic
    data.pop("frequency_penalty", None)
    data.pop("logprobs", None)
    data.pop("n", None)
    data.pop("top_p", None)
    data.pop("presence_penalty", None)
    data.pop("user", None)

    response_json = make_post_request(url, headers, data)
    return convert_anthropic_response_to_chatcompletion(response_json=response_json, inner_thoughts_xml_tag=inner_thoughts_xml_tag)
