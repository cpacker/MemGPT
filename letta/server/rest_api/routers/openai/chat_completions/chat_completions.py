import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException

from letta.schemas.letta_message import FunctionCall, LettaMessage
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_request import ChatCompletionRequest
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice
from letta.schemas.openai.chat_completion_response import Message as CompletionMessage
from letta.schemas.openai.chat_completion_response import UsageStatistics

# TODO this belongs in a controller!
from letta.server.rest_api.routers.v1.agents import send_message_to_agent
from letta.server.rest_api.utils import get_letta_server

if TYPE_CHECKING:
    from letta.server.server import SyncServer

router = APIRouter(prefix="/v1/chat/completions", tags=["chat_completions"])


@router.post("/voice", response_model=ChatCompletionResponse)
async def create_voice_chat_completion(
    completion_request: ChatCompletionRequest = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    """Send a message to a Letta agent via a /chat/completions completion_request
    This is intended to be used by voice providers, such as LiveKit.
    The bearer token will be used to identify the user.
    The 'user' field in the completion_request should be set to the agent ID.
    """
    print(f"GOT REQUEST: {completion_request.model_dump(exclude_none=True)}")

    actor = server.get_user_or_default(user_id=user_id)

    agent_id = completion_request.user
    if agent_id is None:
        raise HTTPException(status_code=400, detail="Must pass agent_id in the 'user' field")

    if not completion_request.stream:
        raise HTTPException(status_code=400, detail="Called voice endpoint, but set streaming to False.")

    messages = completion_request.messages
    messages = [messages[-1]]
    if messages is None:
        raise HTTPException(status_code=400, detail="'messages' field must not be empty")
    if len(messages) > 1:
        raise HTTPException(status_code=400, detail="'messages' field must be a list of length 1")
    if messages[0].role != "user":
        raise HTTPException(status_code=400, detail="'messages[0].role' must be a 'user'")

    # Translate to Message objects
    messages = [Message.from_chat_completions_message(m, completion_request, actor.id) for m in messages]
    print("Starting streaming OpenAI proxy response")
    return await send_message_to_agent(
        server=server,
        agent_id=agent_id,
        user_id=actor.id,
        messages=messages,
        # Turn streaming ON
        stream_steps=True,
        stream_tokens=True,
        # Turn on ChatCompletion mode (eg remaps send_message to content)
        voice_chat_completion_mode=True,
        return_message_object=False,
        include_final_message=False,
    )
    return response


@router.post("", response_model=ChatCompletionResponse)
async def create_chat_completion(
    completion_request: ChatCompletionRequest = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """Send a message to a Letta agent via a /chat/completions completion_request
    The bearer token will be used to identify the user.
    The 'user' field in the completion_request should be set to the agent ID.
    """
    print(f"GOT REQUEST: {completion_request.model_dump(exclude_none=True)}")

    actor = server.get_user_or_default(user_id=user_id)

    agent_id = completion_request.user
    if agent_id is None:
        raise HTTPException(status_code=400, detail="Must pass agent_id in the 'user' field")

    messages = completion_request.messages
    messages = [messages[-1]]
    if messages is None:
        raise HTTPException(status_code=400, detail="'messages' field must not be empty")
    if len(messages) > 1:
        raise HTTPException(status_code=400, detail="'messages' field must be a list of length 1")
    if messages[0].role != "user":
        raise HTTPException(status_code=400, detail="'messages[0].role' must be a 'user'")

    # Translate to Message objects
    messages = [Message.from_chat_completions_message(m, completion_request, actor.id) for m in messages]

    if completion_request.stream:
        print("Starting streaming OpenAI proxy response")

        # TODO(charles) support multimodal parts
        return await send_message_to_agent(
            server=server,
            agent_id=agent_id,
            user_id=actor.id,
            messages=messages,
            # Turn streaming ON
            stream_steps=True,
            stream_tokens=True,
            # Turn on ChatCompletion mode (eg remaps send_message to content)
            voice_chat_completion_mode=True,
            return_message_object=False,
            include_final_message=False,
        )
    else:
        print("Starting non-streaming OpenAI proxy response")

        # TODO(charles) support multimodal parts
        response_messages = await send_message_to_agent(
            server=server,
            agent_id=agent_id,
            user_id=actor.id,
            messages=messages,
            # Turn streaming OFF
            stream_steps=False,
            stream_tokens=False,
            return_message_object=False,
        )
        # print(response_messages)

        # Concatenate all send_message outputs together
        id = ""
        visible_message_str = ""
        created_at = None
        for letta_msg in response_messages.messages:
            assert isinstance(letta_msg, LettaMessage)
            if isinstance(letta_msg, FunctionCall):
                if letta_msg.name and letta_msg.name == "send_message":
                    try:
                        letta_function_call_args = json.loads(letta_msg.arguments)
                        visible_message_str += letta_function_call_args["message"]
                        id = letta_msg.id
                        created_at = letta_msg.date
                    except:
                        print(f"Failed to parse Letta message: {str(letta_msg)}")
                else:
                    print(f"Skipping function_call: {str(letta_msg)}")
            else:
                print(f"Skipping message: {str(letta_msg)}")

        response = ChatCompletionResponse(
            id=id,
            created=created_at if created_at else datetime.now(timezone.utc),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=CompletionMessage(
                        role="assistant",
                        content=visible_message_str,
                    ),
                )
            ],
            # TODO add real usage
            usage=UsageStatistics(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )
        return response
