import asyncio
from datetime import datetime
from functools import partial
from typing import List, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from memgpt.schemas.enums import MessageRole, MessageStreamStatus
from memgpt.schemas.memgpt_message import LegacyMemGPTMessage, MemGPTMessage
from memgpt.schemas.memgpt_request import MemGPTRequest
from memgpt.schemas.memgpt_response import MemGPTResponse
from memgpt.schemas.message import Message
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface, StreamingServerInterface
from memgpt.server.rest_api.utils import sse_async_generator
from memgpt.server.server import SyncServer
from memgpt.utils import deduplicate

router = APIRouter()


# TODO: cpacker should check this file
# TODO: move this into server.py?
async def send_message_to_agent(
    server: SyncServer,
    agent_id: str,
    user_id: str,
    role: MessageRole,
    message: str,
    stream_steps: bool,
    stream_tokens: bool,
    return_message_object: bool,  # Should be True for Python Client, False for REST API
    chat_completion_mode: Optional[bool] = False,
    timestamp: Optional[datetime] = None,
    # related to whether or not we return `MemGPTMessage`s or `Message`s
) -> Union[StreamingResponse, MemGPTResponse]:
    """Split off into a separate function so that it can be imported in the /chat/completion proxy."""
    # TODO: @charles is this the correct way to handle?
    include_final_message = True

    # determine role
    if role == MessageRole.user:
        message_func = server.user_message
    elif role == MessageRole.system:
        message_func = server.system_message
    else:
        raise HTTPException(status_code=500, detail=f"Bad role {role}")

    if not stream_steps and stream_tokens:
        raise HTTPException(status_code=400, detail="stream_steps must be 'true' if stream_tokens is 'true'")

    # For streaming response
    try:

        # TODO: move this logic into server.py

        # Get the generator object off of the agent's streaming interface
        # This will be attached to the POST SSE request used under-the-hood
        memgpt_agent = server._get_or_load_agent(agent_id=agent_id)
        streaming_interface = memgpt_agent.interface
        if not isinstance(streaming_interface, StreamingServerInterface):
            raise ValueError(f"Agent has wrong type of interface: {type(streaming_interface)}")

        # Enable token-streaming within the request if desired
        streaming_interface.streaming_mode = stream_tokens
        # "chatcompletion mode" does some remapping and ignores inner thoughts
        streaming_interface.streaming_chat_completion_mode = chat_completion_mode

        # streaming_interface.allow_assistant_message = stream
        # streaming_interface.function_call_legacy_mode = stream

        # Offload the synchronous message_func to a separate thread
        streaming_interface.stream_start()
        task = asyncio.create_task(
            asyncio.to_thread(message_func, user_id=user_id, agent_id=agent_id, message=message, timestamp=timestamp)
        )

        if stream_steps:
            if return_message_object:
                # TODO implement returning `Message`s in a stream, not just `MemGPTMessage` format
                raise NotImplementedError

            # return a stream
            return StreamingResponse(
                sse_async_generator(streaming_interface.get_generator(), finish_message=include_final_message),
                media_type="text/event-stream",
            )

        else:
            # buffer the stream, then return the list
            generated_stream = []
            async for message in streaming_interface.get_generator():
                assert (
                    isinstance(message, MemGPTMessage)
                    or isinstance(message, LegacyMemGPTMessage)
                    or isinstance(message, MessageStreamStatus)
                ), type(message)
                generated_stream.append(message)
                if message == MessageStreamStatus.done:
                    break

            # Get rid of the stream status messages
            filtered_stream = [d for d in generated_stream if not isinstance(d, MessageStreamStatus)]
            usage = await task

            # By default the stream will be messages of type MemGPTMessage or MemGPTLegacyMessage
            # If we want to convert these to Message, we can use the attached IDs
            # NOTE: we will need to de-duplicate the Messsage IDs though (since Assistant->Inner+Func_Call)
            # TODO: eventually update the interface to use `Message` and `MessageChunk` (new) inside the deque instead
            if return_message_object:
                message_ids = [m.id for m in filtered_stream]
                message_ids = deduplicate(message_ids)
                message_objs = [server.get_agent_message(agent_id=agent_id, message_id=m_id) for m_id in message_ids]
                return MemGPTResponse(messages=message_objs, usage=usage)
            else:
                return MemGPTResponse(messages=filtered_stream, usage=usage)

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e}")


def setup_agents_message_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/messages/context/", tags=["agents"], response_model=List[Message])
    def get_agent_messages_in_context(
        agent_id: str,
        start: int = Query(..., description="Message index to start on (reverse chronological)."),
        count: int = Query(..., description="How many messages to retrieve."),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve the in-context messages of a specific agent. Paginated, provide start and count to iterate.
        """
        interface.clear()
        messages = server.get_agent_messages(agent_id=agent_id, start=start, count=count)
        return messages

    @router.get("/agents/{agent_id}/messages", tags=["agents"], response_model=List[Message])
    def get_agent_messages(
        agent_id: str,
        before: Optional[str] = Query(None, description="Message before which to retrieve the returned messages."),
        limit: int = Query(10, description="Maximum number of messages to retrieve."),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Retrieve message history for an agent.
        """
        interface.clear()
        return server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, before=before, limit=limit, reverse=True)

    @router.post("/agents/{agent_id}/messages", tags=["agents"], response_model=MemGPTResponse)
    async def send_message(
        # background_tasks: BackgroundTasks,
        agent_id: str,
        request: MemGPTRequest = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Process a user message and return the agent's response.

        This endpoint accepts a message from a user and processes it through the agent.
        It can optionally stream the response if 'stream' is set to True.
        """
        # TODO: should this recieve multiple messages? @cpacker
        # TODO: revise to `MemGPTRequest`
        # TODO: support sending multiple messages
        assert len(request.messages) == 1, f"Multiple messages not supported: {request.messages}"
        message = request.messages[0]

        # TODO: what to do with message.name?
        return await send_message_to_agent(
            server=server,
            agent_id=agent_id,
            user_id=user_id,
            role=message.role,
            message=message.text,
            stream_steps=request.stream_steps,
            stream_tokens=request.stream_tokens,
            return_message_object=request.return_message_object,
        )

    return router
