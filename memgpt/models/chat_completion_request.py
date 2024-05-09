from typing import List, Union, Optional, Literal
from pydantic import BaseModel


class SystemMessage(BaseModel):
    content: str
    role: str = "system"
    name: Optional[str] = None


class UserMessage(BaseModel):
    content: Union[str, List[str]]
    role: str = "user"
    name: Optional[str] = None


class AssistantMessage(BaseModel):
    content: Optional[str] = None
    role: str = "assistant"
    name: Optional[str] = None
    tool_calls: Optional[List] = None


ChatMessage = Union[SystemMessage, UserMessage, AssistantMessage]


## tool_choice ##
class FunctionCall(BaseModel):
    name: str


class ToolFunctionChoice(BaseModel):
    # The type of the tool. Currently, only function is supported
    type: Literal["function"] = "function"
    function: FunctionCall


ToolChoice = Union[Literal["none", "auto"], ToolFunctionChoice]


## function_call ##
FunctionCallChoice = Union[Literal["none", "auto"], FunctionCall]
