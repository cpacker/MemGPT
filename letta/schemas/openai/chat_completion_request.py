from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    content: str
    role: str = "system"
    name: Optional[str] = None


class UserMessage(BaseModel):
    content: Union[str, List[str]]
    role: str = "user"
    name: Optional[str] = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    content: Optional[str] = None
    role: str = "assistant"
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseModel):
    content: str
    role: str = "tool"
    tool_call_id: str


ChatMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# TODO: this might not be necessary with the validator
def cast_message_to_subtype(m_dict: dict) -> ChatMessage:
    """Cast a dictionary to one of the individual message types"""
    role = m_dict.get("role")
    if role == "system":
        return SystemMessage(**m_dict)
    elif role == "user":
        return UserMessage(**m_dict)
    elif role == "assistant":
        return AssistantMessage(**m_dict)
    elif role == "tool":
        return ToolMessage(**m_dict)
    else:
        raise ValueError("Unknown message role")


class ResponseFormat(BaseModel):
    type: str = Field(default="text", pattern="^(text|json_object)$")


## tool_choice ##
class FunctionCall(BaseModel):
    name: str


class ToolFunctionChoice(BaseModel):
    # The type of the tool. Currently, only function is supported
    type: Literal["function"] = "function"
    # type: str = Field(default="function", const=True)
    function: FunctionCall


ToolChoice = Union[Literal["none", "auto", "required"], ToolFunctionChoice]


## tools ##
class FunctionSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema for the parameters


class Tool(BaseModel):
    # The type of the tool. Currently, only function is supported
    type: Literal["function"] = "function"
    # type: str = Field(default="function", const=True)
    function: FunctionSchema


## function_call ##
FunctionCallChoice = Union[Literal["none", "auto"], FunctionCall]


class ChatCompletionRequest(BaseModel):
    """https://platform.openai.com/docs/api-reference/chat/create"""

    model: str
    messages: List[ChatMessage]
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    user: Optional[str] = None  # unique ID of the end-user (for monitoring)

    # function-calling related
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None  # "none" means don't call a tool
    # deprecated scheme
    functions: Optional[List[FunctionSchema]] = None
    function_call: Optional[FunctionCallChoice] = None
