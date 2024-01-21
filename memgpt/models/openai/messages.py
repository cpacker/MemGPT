from typing import List, Union, Optional, Dict, Literal
from pydantic import BaseModel, Field, Json
import uuid


class ImageFile(BaseModel):
    type: str = "image_file"
    file_id: str


class Text(BaseModel):
    object: str = "text"
    text: str = Field(..., description="The text content to be processed by the agent.")


class OpenAIMessage(BaseModel):
    id: str = Field(..., description="The unique identifier of the message.")
    object: str = "thread.message"
    created_at: int = Field(..., description="The unix timestamp of when the message was created.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    role: str = Field(..., description="Role of the message sender (either 'user' or 'system')")
    content: List[Union[Text, ImageFile]] = Field(None, description="The message content to be processed by the agent.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    run_id: Optional[str] = Field(None, description="The unique identifier of the run.")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs associated with the message.")
    metadata: Optional[Dict] = Field(None, description="Metadata associated with the message.")


class MessageFile(BaseModel):
    id: str
    object: str = "thread.message.file"
    created_at: int  # unix timestamp


class CreateMessageRequest(BaseModel):
    role: str
    content: str
    file_ids: Optional[List[str]] = []
    metadata: Optional[Dict] = {}


class ModifyMessageRequest(BaseModel):
    metadata: Optional[Dict] = None


class ListMessagesResponse(BaseModel):
    messages: List[OpenAIMessage]
