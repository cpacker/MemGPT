import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, HTTPException

from memgpt.schemas.enums import MessageRole
from memgpt.schemas.memgpt_message import FunctionCall, MemGPTMessage
from memgpt.schemas.openai.chat_completion_request import ChatCompletionRequest
from memgpt.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    Message,
    UsageStatistics,
)

# TODO this belongs in a controller!
from memgpt.server.rest_api.routers.v1.agents import send_message_to_agent
from memgpt.server.rest_api.utils import get_memgpt_server

if TYPE_CHECKING:
    pass

    from memgpt.server.server import SyncServer
    from memgpt.utils import get_utc_time

router = APIRouter(prefix="/v1/chat/completions", tags=["chat_completions"])


@router.post("/", response_model=ChatCompletionResponse)
async def create_chat_completion(
    completion_request: ChatCompletionRequest = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Send a message to a MemGPT agent via a /chat/completions completion_request
    The bearer token will be used to identify the user.
    The 'user' field in the completion_request should be set to the agent ID.
    """
    actor = server.get_current_user()
    agent_id = completion_request.user
    if agent_id is None:
        raise HTTPException(status_code=400, detail="Must pass agent_id in the 'user' field")

    messages = completion_request.messages
    if messages is None:
        raise HTTPException(status_code=400, detail="'messages' field must not be empty")
    if len(messages) > 1:
        raise HTTPException(status_code=400, detail="'messages' field must be a list of length 1")
    if messages[0].role != "user":
        raise HTTPException(status_code=400, detail="'messages[0].role' must be a 'user'")

    input_message = completion_request.messages[0]
    if completion_request.stream:
        print("Starting streaming OpenAI proxy response")

        # TODO(charles) support multimodal parts
        assert isinstance(input_message.content, str)

        return await send_message_to_agent(
            server=server,
            agent_id=agent_id,
            user_id=actor.id,
            role=MessageRole(input_message.role),
            message=input_message.content,
            # Turn streaming ON
            stream_steps=True,
            stream_tokens=True,
            # Turn on ChatCompletion mode (eg remaps send_message to content)
            chat_completion_mode=True,
            return_message_object=False,
        )

    else:
        print("Starting non-streaming OpenAI proxy response")

        # TODO(charles) support multimodal parts
        assert isinstance(input_message.content, str)

        response_messages = await send_message_to_agent(
            server=server,
            agent_id=agent_id,
            user_id=actor.id,
            role=MessageRole(input_message.role),
            message=input_message.content,
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
        for memgpt_msg in response_messages.messages:
            assert isinstance(memgpt_msg, MemGPTMessage)
            if isinstance(memgpt_msg, FunctionCall):
                if memgpt_msg.name and memgpt_msg.name == "send_message":
                    try:
                        memgpt_function_call_args = json.loads(memgpt_msg.arguments)
                        visible_message_str += memgpt_function_call_args["message"]
                        id = memgpt_msg.id
                        created_at = memgpt_msg.date
                    except:
                        print(f"Failed to parse MemGPT message: {str(memgpt_msg)}")
                else:
                    print(f"Skipping function_call: {str(memgpt_msg)}")
            else:
                print(f"Skipping message: {str(memgpt_msg)}")

        response = ChatCompletionResponse(
            id=id,
            created=created_at if created_at else get_utc_time(),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=Message(
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
