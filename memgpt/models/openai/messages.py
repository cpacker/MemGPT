from typing import List, Union, Optional, Dict, Literal
from pydantic import BaseModel, Field, Json
import uuid


class ImageFile(BaseModel):
    type: str = "image_file"
    file_id: str


class Text(BaseModel):
    type: str = "text"
    text: str


class Message(BaseModel):
    id: str
    object: str = "thread.message"
    created_at: int  # unix timestamp
    thread_id: str
    role: str
    content: List[Union[ImageFile, Text]]
    assistant_id: str
    run_id: Optional[str] = None
    file_ids: Optional[List[str]] = []
    metadata: Optional[Dict] = {}


class MessageFile(BaseModel):
    id: str
    object: str = "thread.message.file"
    created_at: int  # unix timestamp
    message_id: str


class CreateMessageRequest(BaseModel):
    role: str
    content: str
    file_ids: Optional[List[str]] = []
    metadata: Optional[Dict] = {}


class ModifyMessageRequest(BaseModel):
    metadata: Optional[Dict] = None


class ListMessagesResponse(BaseModel):
    messages: List[Message]
