from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, field_serializer

# MemGPT API style responses (intended to be easier to use vs getting true Message types)


class BaseMemGPTMessage(BaseModel):
    id: str
    date: datetime

    @field_serializer("date")
    def serialize_datetime(self, dt: datetime, _info):
        return dt.now(timezone.utc).isoformat()


class InternalMonologue(BaseMemGPTMessage):
    """
    {
        "internal_monologue": msg,
        "date": msg_obj.created_at.isoformat() if msg_obj is not None else get_utc_time().isoformat(),
        "id": str(msg_obj.id) if msg_obj is not None else None,
    }
    """

    internal_monologue: str


class FunctionCall(BaseModel):
    name: str
    arguments: str


class FunctionCallMessage(BaseMemGPTMessage):
    """
    {
        "function_call": {
            "name": function_call.function.name,
            "arguments": function_call.function.arguments,
        },
        "id": str(msg_obj.id),
        "date": msg_obj.created_at.isoformat(),
    }
    """

    function_call: FunctionCall


class FunctionReturn(BaseMemGPTMessage):
    """
    {
        "function_return": msg,
        "status": "success" or "error",
        "id": str(msg_obj.id),
        "date": msg_obj.created_at.isoformat(),
    }
    """

    function_return: str
    status: Literal["success", "error"]


MemGPTMessage = Union[InternalMonologue, FunctionCallMessage, FunctionReturn]


# Legacy MemGPT API had an additional type "assistant_message" and the "function_call" was a formatted string


class AssistantMessage(BaseMemGPTMessage):
    assistant_message: str


class LegacyFunctionCallMessage(BaseMemGPTMessage):
    function_call: str


LegacyMemGPTMessage = Union[InternalMonologue, AssistantMessage, LegacyFunctionCallMessage, FunctionReturn]
