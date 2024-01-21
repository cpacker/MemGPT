import asyncio
from fastapi import FastAPI
from asyncio import AbstractEventLoop
from enum import Enum
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Body, HTTPException, Query, Path
from pydantic import BaseModel, Field, constr, validator
from starlette.responses import StreamingResponse

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.static_files import mount_static_files
from memgpt.models.openai.messages import OpenAIMessage, Text
from memgpt.data_types import LLMConfig, EmbeddingConfig
from memgpt.constants import DEFAULT_PRESET

"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""

interface: QueuingInterface = QueuingInterface()
server: SyncServer = SyncServer(default_interface=interface)


# router = APIRouter()
app = FastAPI()


class MessageRoleType(str, Enum):
    user = "user"
    system = "system"


class OpenAIAssistant(BaseModel):
    """Represents an OpenAI assistant (equivalent to MemGPT preset)"""

    id: str = Field(..., description="The unique identifier of the assistant.")
    name: str = Field(..., description="The name of the assistant.")
    object: str = "assistant"
    description: str = Field(..., description="The description of the assistant.")
    created_at: int = Field(..., description="The unix timestamp of when the assistant was created.")
    model: str = Field(..., description="The model used by the assistant.")
    instructions: str = Field(..., description="The instructions for the assistant.")
    tools: List[str] = Field(..., description="The tools used by the assistant.")
    file_ids: List[str] = Field(..., description="List of file IDs associated with the assistant.")
    metadata: dict = Field(..., description="Metadata associated with the assistant.")


class CreateAssistantRequest(BaseModel):
    model: str = Field(..., description="The model to use for the assistant.")
    name: str = Field(..., description="The name of the assistant.")
    description: str = Field(..., description="The description of the assistant.")
    instructions: str = Field(..., description="The instructions for the assistant.")
    tools: List[str] = Field(..., description="The tools used by the assistant.")
    file_ids: List[str] = Field(..., description="List of file IDs associated with the assistant.")
    metadata: dict = Field(..., description="Metadata associated with the assistant.")

    # memgpt-only (not openai)
    embedding_model: str = Field(..., description="The model to use for the assistant.")

    # TODO: remove
    user_id: str = Field(..., description="The unique identifier of the user.")


class OpenAIThread(BaseModel):
    """Represents an OpenAI thread (equivalent to MemGPT agent)"""

    id: str = Field(..., description="The unique identifier of the thread.")
    object: str = "thread"
    created_at: int = Field(..., description="The unix timestamp of when the thread was created.")
    metadata: dict = Field(None, description="Metadata associated with the thread.")


class CreateThreadRequest(BaseModel):
    messages: List[str] = Field(None, description="List of message IDs associated with the thread.")
    metadata: dict = Field(None, description="Metadata associated with the thread.")

    # memgpt-only
    assistant_name: str = Field(..., description="The name of the assistant (i.e. MemGPT preset)")

    # TODO: remove
    user_id: str = Field(..., description="The unique identifier of the user.")


class CreateMessageRequest(BaseModel):
    role: str = Field(..., description="Role of the message sender (either 'user' or 'system')")
    content: str = Field(..., description="The message content to be processed by the agent.")
    file_ids: Optional[List[str]] = Field(..., description="List of file IDs associated with the message.")
    metadata: Optional[dict] = Field(..., description="Metadata associated with the message.")

    # TODO: remove
    user_id: str = Field(..., description="The unique identifier of the user.")


class UserMessageRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier of the user.")
    agent_id: str = Field(..., description="The unique identifier of the agent.")
    message: str = Field(..., description="The message content to be processed by the agent.")
    stream: bool = Field(default=False, description="Flag to determine if the response should be streamed. Set to True for streaming.")
    role: MessageRoleType = Field(default=MessageRoleType.user, description="Role of the message sender (either 'user' or 'system')")


class UserMessageResponse(BaseModel):
    messages: List[dict] = Field(..., description="List of messages generated by the agent in response to the received message.")


class GetAgentMessagesRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier of the user.")
    agent_id: str = Field(..., description="The unique identifier of the agent.")
    start: int = Field(..., description="Message index to start on (reverse chronological).")
    count: int = Field(..., description="How many messages to retrieve.")


class ListMessagesResponse(BaseModel):
    messages: list = Field(..., description="List of message objects.")


# TODO: implement mechanism for creating/authenticating users associated with a bearer token

# create assistant (MemGPT agent)
@app.post("/v1/assistants", tags=["assistants"], response_model=OpenAIAssistant)
def create_assistant(request: CreateAssistantRequest = Body(...)):
    # TODO: create preset
    return OpenAIAssistant(id=DEFAULT_PRESET, name="default_preset")


@app.post("/v1/threads/", tags=["assistants"], response_model=OpenAIThread)
def create_thread(request: CreateThreadRequest = Body(...)):

    print("threads", request)

    # TODO: use requests.description and requests.metadata fields
    # TODO: handle requests.file_ids and requests.tools
    # TODO: eventually allow request to override embedding/llm model

    # create a memgpt agent
    user_id = uuid.UUID(request.user_id)
    agent_state = server.create_agent(
        user_id=user_id,
        agent_config={
            "user_id": user_id,
            "preset": request.assistant_name,
        },
    )

    # TODO: insert messages into recall memory

    return OpenAIThread(
        id=str(agent_state.id),
        created_at=int(agent_state.created_at.timestamp()),
    )


@app.post("/v1/threads/{thread_id}/messages", tags=["assistants"], response_model=OpenAIMessage)
def create_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateMessageRequest = Body(...),
):
    user_id = uuid.UUID(request.user_id)
    agent_id = uuid.UUID(request.thread_id)
    # TODO: need to add a buffer/queue to server and pull on .step()


@app.get("/v1/threads/{thread_id}/messages", tags=["assistants"], response_model=ListMessagesResponse)
def list_messages(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    limit: int = Query(1000, description="How many messages to retrieve."),
    order: str = Query("asc", description="Order of messages to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    user_id: str = Query(..., description="The unique identifier of the user."),  # TODO: remove
):
    after_uuid = uuid.UUID(after) if before else None
    before_uuid = uuid.UUID(before) if before else None
    user_id = uuid.UUID(user_id)
    agent_id = uuid.UUID(thread_id)
    reverse = True if (order == "desc") else False
    cursor, json_messages = server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        limit=limit,
        after=after_uuid,
        before=before_uuid,
        order_by="created_at",
        reverse=reverse,
    )
    print(json_messages[0]["text"])
    # convert to openai style messages
    openai_messages = [
        OpenAIMessage(
            id=str(message["id"]),
            created_at=int(message["created_at"].timestamp()),
            content=[Text(text=message["text"])],
            role=message["role"],
            thread_id=str(message["agent_id"]),
            assistant_id=DEFAULT_PRESET  # TODO: update this
            # file_ids=message.file_ids,
            # metadata=message.metadata,
        )
        for message in json_messages
    ]
    # TODO: cast back to message objects
    return ListMessagesResponse(messages=openai_messages)
