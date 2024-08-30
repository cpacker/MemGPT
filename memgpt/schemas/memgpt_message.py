import json
from datetime import datetime, timezone
from typing import Literal, Optional, Union

from pydantic import BaseModel, field_serializer, field_validator

# MemGPT API style responses (intended to be easier to use vs getting true Message types)


class BaseMemGPTMessage(BaseModel):
    id: str
    date: datetime

    @field_serializer("date")
    def serialize_datetime(self, dt: datetime, _info):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Remove microseconds since it seems like we're inconsistent with getting them
        # TODO figure out why we don't always get microseconds (get_utc_time() does)
        return dt.isoformat(timespec="seconds")


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


class FunctionCallDelta(BaseModel):
    name: Optional[str]
    arguments: Optional[str]

    # NOTE: this is a workaround to exclude None values from the JSON dump,
    # since the OpenAI style of returning chunks doesn't include keys with null values
    def model_dump(self, *args, **kwargs):
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)

    def json(self, *args, **kwargs):
        return json.dumps(self.model_dump(exclude_none=True), *args, **kwargs)


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

    function_call: Union[FunctionCall, FunctionCallDelta]

    # NOTE: this is required for the FunctionCallDelta exclude_none to work correctly
    def model_dump(self, *args, **kwargs):
        kwargs["exclude_none"] = True
        data = super().model_dump(*args, **kwargs)
        if isinstance(data["function_call"], dict):
            data["function_call"] = {k: v for k, v in data["function_call"].items() if v is not None}
        return data

    class Config:
        json_encoders = {
            FunctionCallDelta: lambda v: v.model_dump(exclude_none=True),
            FunctionCall: lambda v: v.model_dump(exclude_none=True),
        }

    # NOTE: this is required to cast dicts into FunctionCallMessage objects
    # Without this extra validator, Pydantic will throw an error if 'name' or 'arguments' are None
    # (instead of properly casting to FunctionCallDelta instead of FunctionCall)
    @field_validator("function_call", mode="before")
    @classmethod
    def validate_function_call(cls, v):
        if isinstance(v, dict):
            if "name" in v and "arguments" in v:
                return FunctionCall(name=v["name"], arguments=v["arguments"])
            elif "name" in v or "arguments" in v:
                return FunctionCallDelta(name=v.get("name"), arguments=v.get("arguments"))
            else:
                raise ValueError("function_call must contain either 'name' or 'arguments'")
        return v


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
