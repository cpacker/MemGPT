import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse

from letta.constants import DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG
from letta.schemas.agent import AgentState, CreateAgent, UpdateAgentState
from letta.schemas.enums import MessageStreamStatus
from letta.schemas.letta_message import (
    LegacyLettaMessage,
    LettaMessage,
    LettaMessageUnion,
)
from letta.schemas.letta_request import LettaRequest
from letta.schemas.letta_response import LettaResponse
from letta.schemas.memory import (
    ArchivalMemorySummary,
    BasicBlockMemory,
    ContextWindowOverview,
    CreateArchivalMemory,
    Memory,
    RecallMemorySummary,
)
from letta.schemas.message import Message, MessageCreate, UpdateMessage
from letta.schemas.passage import Passage
from letta.schemas.source import Source
from letta.schemas.tool import Tool
from letta.server.rest_api.interface import StreamingServerInterface
from letta.server.rest_api.utils import get_letta_server, sse_async_generator
from letta.server.server import SyncServer
from letta.utils import deduplicate

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally


router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/", response_model=List[AgentState], operation_id="list_agents")
def list_agents(
    name: Optional[str] = Query(None, description="Name of the agent"),
    tags: Optional[List[str]] = Query(None, description="List of tags to filter agents by"),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    List all agents associated with a given user.
    This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
    """
    actor = server.get_user_or_default(user_id=user_id)

    agents = server.list_agents(user_id=actor.id, tags=tags)
    # TODO: move this logic to the ORM
    if name:
        agents = [a for a in agents if a.name == name]
    return agents


@router.get("/{agent_id}/context", response_model=ContextWindowOverview, operation_id="get_agent_context_window")
def get_agent_context_window(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Retrieve the context window of a specific agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.get_agent_context_window(user_id=actor.id, agent_id=agent_id)


@router.post("/", response_model=AgentState, operation_id="create_agent")
def create_agent(
    agent: CreateAgent = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Create a new agent with the specified configuration.
    """
    actor = server.get_user_or_default(user_id=user_id)
    agent.user_id = actor.id
    # TODO: sarah make general
    # TODO: eventually remove this
    assert agent.memory is not None  # TODO: dont force this, can be None (use default human/person)
    blocks = agent.memory.get_blocks()
    agent.memory = BasicBlockMemory(blocks=blocks)

    return server.create_agent(agent, actor=actor)


@router.patch("/{agent_id}", response_model=AgentState, operation_id="update_agent")
def update_agent(
    agent_id: str,
    update_agent: UpdateAgentState = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """Update an exsiting agent"""
    actor = server.get_user_or_default(user_id=user_id)
    return server.update_agent(update_agent, actor=actor)


@router.get("/{agent_id}/tools", response_model=List[Tool], operation_id="get_tools_from_agent")
def get_tools_from_agent(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """Get tools from an existing agent"""
    actor = server.get_user_or_default(user_id=user_id)
    return server.get_tools_from_agent(agent_id=agent_id, user_id=actor.id)


@router.patch("/{agent_id}/add-tool/{tool_id}", response_model=AgentState, operation_id="add_tool_to_agent")
def add_tool_to_agent(
    agent_id: str,
    tool_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """Add tools to an existing agent"""
    actor = server.get_user_or_default(user_id=user_id)
    return server.add_tool_to_agent(agent_id=agent_id, tool_id=tool_id, user_id=actor.id)


@router.patch("/{agent_id}/remove-tool/{tool_id}", response_model=AgentState, operation_id="remove_tool_from_agent")
def remove_tool_from_agent(
    agent_id: str,
    tool_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """Add tools to an existing agent"""
    actor = server.get_user_or_default(user_id=user_id)
    return server.remove_tool_from_agent(agent_id=agent_id, tool_id=tool_id, user_id=actor.id)


@router.get("/{agent_id}", response_model=AgentState, operation_id="get_agent")
def get_agent_state(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Get the state of the agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    if not server.ms.get_agent(user_id=actor.id, agent_id=agent_id):
        # agent does not exist
        raise HTTPException(status_code=404, detail=f"Agent agent_id={agent_id} not found.")

    return server.get_agent_state(user_id=actor.id, agent_id=agent_id)


@router.delete("/{agent_id}", response_model=None, operation_id="delete_agent")
def delete_agent(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Delete an agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.delete_agent(user_id=actor.id, agent_id=agent_id)


@router.get("/{agent_id}/sources", response_model=List[Source], operation_id="get_agent_sources")
def get_agent_sources(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Get the sources associated with an agent.
    """

    return server.list_attached_sources(agent_id)


@router.get("/{agent_id}/memory/messages", response_model=List[Message], operation_id="list_agent_in_context_messages")
def get_agent_in_context_messages(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Retrieve the messages in the context of a specific agent.
    """

    return server.get_in_context_messages(agent_id=agent_id)


@router.get("/{agent_id}/memory", response_model=Memory, operation_id="get_agent_memory")
def get_agent_memory(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Retrieve the memory state of a specific agent.
    This endpoint fetches the current memory state of the agent identified by the user ID and agent ID.
    """

    return server.get_agent_memory(agent_id=agent_id)


@router.patch("/{agent_id}/memory", response_model=Memory, operation_id="update_agent_memory")
def update_agent_memory(
    agent_id: str,
    request: Dict = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Update the core memory of a specific agent.
    This endpoint accepts new memory contents (human and persona) and updates the core memory of the agent identified by the user ID and agent ID.
    """
    actor = server.get_user_or_default(user_id=user_id)

    memory = server.update_agent_core_memory(user_id=actor.id, agent_id=agent_id, new_memory_contents=request)
    return memory


@router.get("/{agent_id}/memory/recall", response_model=RecallMemorySummary, operation_id="get_agent_recall_memory_summary")
def get_agent_recall_memory_summary(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Retrieve the summary of the recall memory of a specific agent.
    """

    return server.get_recall_memory_summary(agent_id=agent_id)


@router.get("/{agent_id}/memory/archival", response_model=ArchivalMemorySummary, operation_id="get_agent_archival_memory_summary")
def get_agent_archival_memory_summary(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Retrieve the summary of the archival memory of a specific agent.
    """

    return server.get_archival_memory_summary(agent_id=agent_id)


@router.get("/{agent_id}/archival", response_model=List[Passage], operation_id="list_agent_archival_memory")
def get_agent_archival_memory(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    after: Optional[int] = Query(None, description="Unique ID of the memory to start the query range at."),
    before: Optional[int] = Query(None, description="Unique ID of the memory to end the query range at."),
    limit: Optional[int] = Query(None, description="How many results to include in the response."),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Retrieve the memories in an agent's archival memory store (paginated query).
    """
    actor = server.get_user_or_default(user_id=user_id)

    # TODO need to add support for non-postgres here
    # chroma will throw:
    #     raise ValueError("Cannot run get_all_cursor with chroma")

    return server.get_agent_archival_cursor(
        user_id=actor.id,
        agent_id=agent_id,
        after=after,
        before=before,
        limit=limit,
    )


@router.post("/{agent_id}/archival", response_model=List[Passage], operation_id="create_agent_archival_memory")
def insert_agent_archival_memory(
    agent_id: str,
    request: CreateArchivalMemory = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Insert a memory into an agent's archival memory store.
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.insert_archival_memory(user_id=actor.id, agent_id=agent_id, memory_contents=request.text)


# TODO(ethan): query or path parameter for memory_id?
# @router.delete("/{agent_id}/archival")
@router.delete("/{agent_id}/archival/{memory_id}", response_model=None, operation_id="delete_agent_archival_memory")
def delete_agent_archival_memory(
    agent_id: str,
    memory_id: str,
    # memory_id: str = Query(..., description="Unique ID of the memory to be deleted."),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Delete a memory from an agent's archival memory store.
    """
    actor = server.get_user_or_default(user_id=user_id)

    server.delete_archival_memory(user_id=actor.id, agent_id=agent_id, memory_id=memory_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Memory id={memory_id} successfully deleted"})


@router.get("/{agent_id}/messages", response_model=Union[List[Message], List[LettaMessageUnion]], operation_id="list_agent_messages")
def get_agent_messages(
    agent_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    before: Optional[str] = Query(None, description="Message before which to retrieve the returned messages."),
    limit: int = Query(10, description="Maximum number of messages to retrieve."),
    msg_object: bool = Query(False, description="If true, returns Message objects. If false, return LettaMessage objects."),
    # Flags to support the use of AssistantMessage message types
    use_assistant_message: bool = Query(
        False,
        description="[Only applicable if msg_object is False] If true, returns AssistantMessage objects when the agent calls a designated message tool. If false, return FunctionCallMessage objects for all tool calls.",
    ),
    assistant_message_function_name: str = Query(
        DEFAULT_MESSAGE_TOOL,
        description="[Only applicable if use_assistant_message is True] The name of the designated message tool.",
    ),
    assistant_message_function_kwarg: str = Query(
        DEFAULT_MESSAGE_TOOL_KWARG,
        description="[Only applicable if use_assistant_message is True] The name of the message argument in the designated message tool.",
    ),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Retrieve message history for an agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.get_agent_recall_cursor(
        user_id=actor.id,
        agent_id=agent_id,
        before=before,
        limit=limit,
        reverse=True,
        return_message_object=msg_object,
        use_assistant_message=use_assistant_message,
        assistant_message_function_name=assistant_message_function_name,
        assistant_message_function_kwarg=assistant_message_function_kwarg,
    )


@router.patch("/{agent_id}/messages/{message_id}", response_model=Message, operation_id="update_agent_message")
def update_message(
    agent_id: str,
    message_id: str,
    request: UpdateMessage = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Update the details of a message associated with an agent.
    """
    assert request.id == message_id, f"Message ID mismatch: {request.id} != {message_id}"
    return server.update_agent_message(agent_id=agent_id, request=request)


@router.post(
    "/{agent_id}/messages",
    response_model=None,
    operation_id="create_agent_message",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {"$ref": "#/components/schemas/LettaResponse"},  # Use model_json_schema() instead of model directly
                "text/event-stream": {"description": "Server-Sent Events stream"},
            },
        }
    },
)
async def send_message(
    agent_id: str,
    server: SyncServer = Depends(get_letta_server),
    request: LettaRequest = Body(...),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Process a user message and return the agent's response.
    This endpoint accepts a message from a user and processes it through the agent.
    It can optionally stream the response if 'stream_steps' or 'stream_tokens' is set to True.
    """
    actor = server.get_user_or_default(user_id=user_id)

    result = await send_message_to_agent(
        server=server,
        agent_id=agent_id,
        user_id=actor.id,
        messages=request.messages,
        stream_steps=request.stream_steps,
        stream_tokens=request.stream_tokens,
        return_message_object=request.return_message_object,
        # Support for AssistantMessage
        use_assistant_message=request.use_assistant_message,
        assistant_message_function_name=request.assistant_message_function_name,
        assistant_message_function_kwarg=request.assistant_message_function_kwarg,
    )
    return result


# TODO: move this into server.py?
async def send_message_to_agent(
    server: SyncServer,
    agent_id: str,
    user_id: str,
    # role: MessageRole,
    messages: Union[List[Message], List[MessageCreate]],
    stream_steps: bool,
    stream_tokens: bool,
    # related to whether or not we return `LettaMessage`s or `Message`s
    return_message_object: bool,  # Should be True for Python Client, False for REST API
    chat_completion_mode: bool = False,
    timestamp: Optional[datetime] = None,
    # Support for AssistantMessage
    use_assistant_message: bool = False,
    assistant_message_function_name: str = DEFAULT_MESSAGE_TOOL,
    assistant_message_function_kwarg: str = DEFAULT_MESSAGE_TOOL_KWARG,
) -> Union[StreamingResponse, LettaResponse]:
    """Split off into a separate function so that it can be imported in the /chat/completion proxy."""

    # TODO: @charles is this the correct way to handle?
    include_final_message = True

    if not stream_steps and stream_tokens:
        raise HTTPException(status_code=400, detail="stream_steps must be 'true' if stream_tokens is 'true'")

    # For streaming response
    try:

        # TODO: move this logic into server.py

        # Get the generator object off of the agent's streaming interface
        # This will be attached to the POST SSE request used under-the-hood
        letta_agent = server._get_or_load_agent(agent_id=agent_id)

        # Disable token streaming if not OpenAI
        # TODO: cleanup this logic
        llm_config = letta_agent.agent_state.llm_config
        if llm_config.model_endpoint_type != "openai" or "inference.memgpt.ai" in llm_config.model_endpoint:
            print("Warning: token streaming is only supported for OpenAI models. Setting to False.")
            stream_tokens = False

        # Create a new interface per request
        letta_agent.interface = StreamingServerInterface()
        streaming_interface = letta_agent.interface
        if not isinstance(streaming_interface, StreamingServerInterface):
            raise ValueError(f"Agent has wrong type of interface: {type(streaming_interface)}")

        # Enable token-streaming within the request if desired
        streaming_interface.streaming_mode = stream_tokens
        # "chatcompletion mode" does some remapping and ignores inner thoughts
        streaming_interface.streaming_chat_completion_mode = chat_completion_mode

        # streaming_interface.allow_assistant_message = stream
        # streaming_interface.function_call_legacy_mode = stream

        # Allow AssistantMessage is desired by client
        streaming_interface.use_assistant_message = use_assistant_message
        streaming_interface.assistant_message_function_name = assistant_message_function_name
        streaming_interface.assistant_message_function_kwarg = assistant_message_function_kwarg

        # Related to JSON buffer reader
        streaming_interface.inner_thoughts_in_kwargs = (
            llm_config.put_inner_thoughts_in_kwargs if llm_config.put_inner_thoughts_in_kwargs is not None else False
        )

        # Offload the synchronous message_func to a separate thread
        streaming_interface.stream_start()
        task = asyncio.create_task(
            asyncio.to_thread(
                server.send_messages,
                user_id=user_id,
                agent_id=agent_id,
                messages=messages,
            )
        )

        if stream_steps:
            if return_message_object:
                # TODO implement returning `Message`s in a stream, not just `LettaMessage` format
                raise NotImplementedError

            # return a stream
            return StreamingResponse(
                sse_async_generator(
                    streaming_interface.get_generator(),
                    usage_task=task,
                    finish_message=include_final_message,
                ),
                media_type="text/event-stream",
            )

        else:
            # buffer the stream, then return the list
            generated_stream = []
            async for message in streaming_interface.get_generator():
                assert (
                    isinstance(message, LettaMessage) or isinstance(message, LegacyLettaMessage) or isinstance(message, MessageStreamStatus)
                ), type(message)
                generated_stream.append(message)
                if message == MessageStreamStatus.done:
                    break

            # Get rid of the stream status messages
            filtered_stream = [d for d in generated_stream if not isinstance(d, MessageStreamStatus)]
            usage = await task

            # By default the stream will be messages of type LettaMessage or LettaLegacyMessage
            # If we want to convert these to Message, we can use the attached IDs
            # NOTE: we will need to de-duplicate the Messsage IDs though (since Assistant->Inner+Func_Call)
            # TODO: eventually update the interface to use `Message` and `MessageChunk` (new) inside the deque instead
            if return_message_object:
                message_ids = [m.id for m in filtered_stream]
                message_ids = deduplicate(message_ids)
                message_objs = [server.get_agent_message(agent_id=agent_id, message_id=m_id) for m_id in message_ids]
                message_objs = [m for m in message_objs if m is not None]
                return LettaResponse(messages=message_objs, usage=usage)
            else:
                return LettaResponse(messages=filtered_stream, usage=usage)

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e}")
