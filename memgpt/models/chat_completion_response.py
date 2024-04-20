import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel

# class ToolCallFunction(BaseModel):
#     name: str
#     arguments: str


class FunctionCall(BaseModel):
    arguments: str
    name: str


class ToolCall(BaseModel):
    id: str
    # "Currently, only function is supported"
    type: Literal["function"] = "function"
    # function: ToolCallFunction
    function: FunctionCall


class LogProbToken(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]]


class MessageContentLogProb(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: Optional[List[LogProbToken]]


class Message(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    role: str
    function_call: Optional[FunctionCall] = None  # Deprecated


class Choice(BaseModel):
    finish_reason: str
    index: int
    message: Message
    logprobs: Optional[Dict[str, Union[List[MessageContentLogProb], None]]] = None


class UsageStatistics(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """https://platform.openai.com/docs/api-reference/chat/object"""

    id: str
    choices: List[Choice]
    created: datetime.datetime
    model: Optional[str] = None  # NOTE: this is not consistent with OpenAI API standard, however is necessary to support local LLMs
    # system_fingerprint: str  # docs say this is mandatory, but in reality API returns None
    system_fingerprint: Optional[str] = None
    # object: str = Field(default="chat.completion")
    object: Literal["chat.completion"] = "chat.completion"
    usage: UsageStatistics


class FunctionCallDelta(BaseModel):
    # arguments: Optional[str] = None
    name: Optional[str] = None
    arguments: str
    # name: str


class ToolCallDelta(BaseModel):
    index: int
    id: Optional[str] = None
    # "Currently, only function is supported"
    type: Literal["function"] = "function"
    # function: ToolCallFunction
    function: Optional[FunctionCallDelta] = None


class MessageDelta(BaseModel):
    """Partial delta stream of a Message

    Example ChunkResponse:
    {
        'id': 'chatcmpl-9EOCkKdicNo1tiL1956kPvCnL2lLS',
        'object': 'chat.completion.chunk',
        'created': 1713216662,
        'model': 'gpt-4-0613',
        'system_fingerprint': None,
        'choices': [{
            'index': 0,
            'delta': {'content': 'User'},
            'logprobs': None,
            'finish_reason': None
        }]
    }
    """

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallDelta]] = None
    # role: Optional[str] = None
    function_call: Optional[FunctionCallDelta] = None  # Deprecated


class ChunkChoice(BaseModel):
    finish_reason: Optional[str] = None  # NOTE: when streaming will be null
    index: int
    delta: MessageDelta
    logprobs: Optional[Dict[str, Union[List[MessageContentLogProb], None]]] = None


class ChatCompletionChunkResponse(BaseModel):
    """https://platform.openai.com/docs/api-reference/chat/streaming"""

    id: str
    choices: List[ChunkChoice]
    created: datetime.datetime
    model: str
    # system_fingerprint: str  # docs say this is mandatory, but in reality API returns None
    system_fingerprint: Optional[str] = None
    # object: str = Field(default="chat.completion")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
