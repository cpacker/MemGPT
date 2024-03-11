import asyncio
import json
import uuid
from asyncio import AbstractEventLoop
from enum import Enum
from functools import partial
from typing import List

from fastapi import APIRouter, Body, HTTPException, Query, Depends
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
    agent_id: str = Field(..., description="The unique identifier of the agent.")
    message: str = Field(..., description="The message content to be processed by the agent.")
    stream: bool = Field(default=False, description="Flag to determine if the response should be streamed. Set to True for streaming.")
    role: MessageRoleType = Field(default=MessageRoleType.user, description="Role of the message sender (either 'user' or 'system')")


class UserMessageResponse(BaseModel):
    messages: List[dict] = Field(..., description="List of messages generated by the agent in response to the received message.")


class GetAgentMessagesRequest(BaseModel):
    agent_id: str = Field(..., description="The unique identifier of the agent.")
    start: int = Field(..., description="Message index to start on (reverse chronological).")
    count: int = Field(..., description="How many messages to retrieve.")


class GetAgentMessagesResponse(BaseModel):
    messages: list = Field(..., description="List of message objects.")


def setup_agents_message_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/message", tags=["agents"], response_model=GetAgentMessagesResponse)
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
                    await asyncio.create_task(message_func(user_id=user_id, agent_id=agent_id, message=request.message))
                else:

                    def handle_exception(exception_loop: AbstractEventLoop, context):
                        # context["message"] will always be there; but context["exception"] may not
                        error = context.get("exception") or context["message"]
                        print(f"handling asyncio exception {context}")
                        interface.error(str(error))

                    # Run the synchronous function in a thread pool
                    loop = asyncio.get_event_loop()
                    loop.set_exception_handler(handle_exception)
                    loop.run_in_executor(None, message_func, user_id, agent_id, request.message)

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
