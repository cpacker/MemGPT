import asyncio
import json
import uuid
from asyncio import AbstractEventLoop
from datetime import datetime
from enum import Enum
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from memgpt.constants import JSON_ENSURE_ASCII
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class MessageRoleType(str, Enum):
    user = "user"
    system = "system"


class UserMessageRequest(BaseModel):
    message: str = Field(..., description="The message content to be processed by the agent.")
    name: str = Field(default="user", description="Name of the message request sender")
    stream: bool = Field(default=False, description="Flag to determine if the response should be streamed. Set to True for streaming.")
    role: MessageRoleType = Field(default=MessageRoleType.user, description="Role of the message sender (either 'user' or 'system')")
    timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp to tag the message with (in ISO format). If null, timestamp will be created server-side on receipt of message.",
    )

    # @validator("timestamp", pre=True, always=True)
    # def validate_timestamp(cls, value: Optional[datetime]) -> Optional[datetime]:
    #    if value is None:
    #        return value  # If the timestamp is None, just return None, implying default handling to set server-side

    #    if not isinstance(value, datetime):
    #        raise TypeError("Timestamp must be a datetime object with timezone information.")

    #    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
    #        raise ValueError("Timestamp must be timezone-aware.")

    #    # Convert timestamp to UTC if it's not already in UTC
    #    if value.tzinfo.utcoffset(value) != timezone.utc.utcoffset(value):
    #        value = value.astimezone(timezone.utc)

    #    return value


class UserMessageResponse(BaseModel):
    messages: List[dict] = Field(..., description="List of messages generated by the agent in response to the received message.")


class GetAgentMessagesRequest(BaseModel):
    start: int = Field(..., description="Message index to start on (reverse chronological).")
    count: int = Field(..., description="How many messages to retrieve.")


class GetAgentMessagesCursorRequest(BaseModel):
    before: Optional[uuid.UUID] = Field(..., description="Message before which to retrieve the returned messages.")
    limit: int = Field(..., description="Maximum number of messages to retrieve.")


class GetAgentMessagesResponse(BaseModel):
    messages: list = Field(..., description="List of message objects.")


def setup_agents_message_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/messages", tags=["agents"], response_model=GetAgentMessagesResponse)
    def get_agent_messages(
        agent_id: uuid.UUID,
        start: int = Query(..., description="Message index to start on (reverse chronological)."),
        count: int = Query(..., description="How many messages to retrieve."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the in-context messages of a specific agent. Paginated, provide start and count to iterate.
        """
        # Validate with the Pydantic model (optional)
        request = GetAgentMessagesRequest(agent_id=agent_id, start=start, count=count)
        # agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        interface.clear()
        messages = server.get_agent_messages(user_id=user_id, agent_id=agent_id, start=request.start, count=request.count)
        return GetAgentMessagesResponse(messages=messages)

    @router.get("/agents/{agent_id}/messages-cursor", tags=["agents"], response_model=GetAgentMessagesResponse)
    def get_agent_messages_cursor(
        agent_id: uuid.UUID,
        before: Optional[uuid.UUID] = Query(None, description="Message before which to retrieve the returned messages."),
        limit: int = Query(10, description="Maximum number of messages to retrieve."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the in-context messages of a specific agent. Paginated, provide start and count to iterate.
        """
        # Validate with the Pydantic model (optional)
        request = GetAgentMessagesCursorRequest(agent_id=agent_id, before=before, limit=limit)

        interface.clear()
        [_, messages] = server.get_agent_recall_cursor(
            user_id=user_id, agent_id=agent_id, before=request.before, limit=request.limit, reverse=True
        )
        # print("====> messages-cursor DEBUG")
        # for i, msg in enumerate(messages):
        # print(f"message {i+1}/{len(messages)}")
        # print(f"UTC created-at: {msg.created_at.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}")
        # print(f"ISO format string: {msg['created_at']}")
        # print(msg)
        return GetAgentMessagesResponse(messages=messages)

    @router.post("/agents/{agent_id}/messages", tags=["agents"], response_model=UserMessageResponse)
    async def send_message(
        agent_id: uuid.UUID,
        request: UserMessageRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Process a user message and return the agent's response.

        This endpoint accepts a message from a user and processes it through the agent.
        It can optionally stream the response if 'stream' is set to True.
        """
        # agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

        if request.role == "user" or request.role is None:
            message_func = server.user_message
        elif request.role == "system":
            message_func = server.system_message
        else:
            raise HTTPException(status_code=500, detail=f"Bad role {request.role}")

        if request.stream:
            # For streaming response
            try:
                # Start the generation process (similar to the non-streaming case)
                # This should be a non-blocking call or run in a background task
                # Check if server.user_message is an async function
                if asyncio.iscoroutinefunction(message_func):
                    # Start the async task
                    await asyncio.create_task(
                        message_func(
                            user_id=user_id,
                            agent_id=agent_id,
                            message=request.message,
                            timestamp=request.timestamp,
                        )
                    )
                else:

                    def handle_exception(exception_loop: AbstractEventLoop, context):
                        # context["message"] will always be there; but context["exception"] may not
                        error = context.get("exception") or context["message"]
                        print(f"handling asyncio exception {context}")
                        interface.error(str(error))

                    # Run the synchronous function in a thread pool
                    loop = asyncio.get_event_loop()
                    loop.set_exception_handler(handle_exception)
                    loop.run_in_executor(
                        None,
                        message_func,
                        user_id,
                        agent_id,
                        request.message,
                        request.timestamp,
                    )

                async def formatted_message_generator():
                    async for message in interface.message_generator():
                        formatted_message = f"data: {json.dumps(message, ensure_ascii=JSON_ENSURE_ASCII)}\n\n"
                        yield formatted_message
                        await asyncio.sleep(1)

                # Return the streaming response using the generator
                return StreamingResponse(formatted_message_generator(), media_type="text/event-stream")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{e}")

        else:
            interface.clear()
            try:
                message_func(user_id=user_id, agent_id=agent_id, message=request.message)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return UserMessageResponse(messages=interface.to_list())

    return router
