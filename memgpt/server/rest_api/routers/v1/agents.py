from typing import Union, Optional, List, Dict
from datetime import datetime
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse

from memgpt.settings import settings
from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.schemas.agent import AgentState, UpdateAgentState
from memgpt.schemas.message import Message
from memgpt.schemas.memory import Memory, RecallMemorySummary, ArchivalMemorySummary
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.llm_config import LLMConfig
from memgpt.server.schemas.agents import AgentCommandResponse, GetAgentResponse, AgentRenameRequest, CreateAgentRequest, CreateAgentResponse, GetAgentMemoryResponse, UpdateAgentMemoryRequest, UpdateAgentMemoryResponse, GetAgentArchivalMemoryResponse, ArchivalMemoryObject, InsertAgentArchivalMemoryRequest, InsertAgentArchivalMemoryResponse, UserMessageRequest, UserMessageResponse, GetAgentMessagesRequest, GetAgentMessagesResponse, GetAgentMessagesCursorRequest
from memgpt.server.rest_api.interface import StreamingServerInterface, QueuingInterface

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally
from uuid import UUID
from memgpt.orm.user import User
from memgpt.server.server import SyncServer
from memgpt.server.schemas.agents import AgentCommandRequest

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/", response_model=List[AgentState])
def list_agents(
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all agents associated with a given user.

    This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
    """
    actor = server.get_current_user()
    interface.clear()
    return server.list_agents(user_id=actor._id)

@router.post("/", response_model=AgentState)
def create_agent(
    agent: CreateAgentRequest,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Create a new agent with the specified configuration.
    """
    actor = server.get_current_user()
    interface.clear()

    return server.create_agent(agent, user_id=actor.id)

@router.post("/{agent_id}", tags=["agents"], response_model=AgentState)
def update_agent(
    agent_id: str,
    update_agent: UpdateAgentState = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """Update an exsiting agent"""
    interface.clear()
    update_agent.id = agent_id
    actor = server.get_current_user()
    return server.update_agent(update_agent, user_id=actor.id)

@router.get("/{agent_id}", response_model=AgentState)
def get_agent_state(
    agent_id: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get the state of the agent.
    """

    interface.clear()
    actor = server.get_current_user()
    if not server.ms.get_agent(user_id=actor.id, agent_id=agent_id):
        # agent does not exist
        raise HTTPException(status_code=404, detail=f"Agent agent_id={agent_id} not found.")

    return server.get_agent_state(user_id=actor.id, agent_id=agent_id)


@router.delete("/{agent_id}")
def delete_agent(
    agent_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """
    Delete an agent.
    """
    interface.clear()
    actor = server.get_current_user()
    return server.delete_agent(user_id=actor.id, agent_id=agent_id)

@router.get("/{agent_id}/memory/messages", response_model=List[Message])
def get_agent_in_context_messages(
    agent_id: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Retrieve the messages in the context of a specific agent.
    """
    interface.clear()
    return server.get_in_context_messages(agent_id=agent_id)


@router.get("/{agent_id}/memory", response_model=Memory)
def get_agent_memory(
    agent_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """
    Retrieve the memory state of a specific agent.

    This endpoint fetches the current memory state of the agent identified by the user ID and agent ID.
    """
    interface.clear()
    return server.get_agent_memory(agent_id=agent_id)

@router.post("/{agent_id}/memory", response_model=Memory)
def update_agent_memory(
    agent_id: str,
    request: Dict = Body(...),
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """
    Update the core memory of a specific agent.

    This endpoint accepts new memory contents (human and persona) and updates the core memory of the agent identified by the user ID and agent ID.
    """
    interface.clear()
    actor = server.get_current_user()
    memory = server.update_agent_core_memory(user_id=actor.id, agent_id=agent_id, new_memory_contents=request)
    return memory

@router.get("/{agent_id}/memory/recall", response_model=RecallMemorySummary)
def get_agent_recall_memory_summary(
    agent_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """
    Retrieve the summary of the recall memory of a specific agent.
    """
    interface.clear()
    return server.get_recall_memory_summary(agent_id=agent_id)

@router.get("/{agent_id}/memory/archival", response_model=ArchivalMemorySummary)
def get_agent_archival_memory_summary(
    agent_id: str,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Retrieve the summary of the archival memory of a specific agent.
    """
    interface.clear()
    return server.get_archival_memory_summary(agent_id=agent_id)



@router.get("/{agent_id}/archival", response_model=List[Passage])
def get_agent_archival_memory(
    agent_id: "str",
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
    after: Optional[int] = Query(None, description="Unique ID of the memory to start the query range at."),
    before: Optional[int] = Query(None, description="Unique ID of the memory to end the query range at."),
    limit: Optional[int] = Query(None, description="How many results to include in the response."),

):
    """
    Retrieve the memories in an agent's archival memory store (paginated query).
    """
    actor = server.get_current_user()
    interface.clear()
    # TODO need to add support for non-postgres here
    # chroma will throw:
    #     raise ValueError("Cannot run get_all_cursor with chroma")
    interface.clear()
    return server.get_agent_archival_cursor(
        user_id=actor.id,
        agent_id=agent_id,
        after=after,
        before=before,
        limit=limit,
    )

@router.post("/{agent_id}/archival", response_model=List[Passage])
def insert_agent_archival_memory(
    agent_id: "str",
    request: InsertAgentArchivalMemoryRequest = Body(...),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Insert a memory into an agent's archival memory store.
    """
    actor = server.get_current_user()
    interface.clear()
    return server.insert_archival_memory(user_id=actor.id, agent_id=agent_id, memory_contents=request.content)

@router.delete("/{agent_id}/archival")
def delete_agent_archival_memory(
    agent_id: "str",
    memory_id: str = Query(..., description="Unique ID of the memory to be deleted."),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Delete a memory from an agent's archival memory store.
    """
    actor = server.get_current_user()
    interface.clear()
    server.delete_archival_memory(user_id=actor.id, agent_id=agent_id, memory_id=memory_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Memory id={memory_id} successfully deleted"})


@router.get("/agents/{agent_id}/messages", tags=["agents"], response_model=GetAgentMessagesResponse)
def get_agent_messages(
    agent_id: "UUID",
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
    start: int = Query(..., description="Message index to start on (reverse chronological)."),
    count: int = Query(..., description="How many messages to retrieve."),
):
    """
    Retrieve the in-context messages of a specific agent. Paginated, provide start and count to iterate.
    """
    # Validate with the Pydantic model (optional)
    actor = server.get_current_user()
    request = GetAgentMessagesRequest(agent_id=agent_id, start=start, count=count)

    interface.clear()
    messages = server.get_agent_messages(user_id=actor._id, agent_id=agent_id, start=request.start, count=request.count)
    return GetAgentMessagesResponse(messages=messages)

@router.get("/{agent_id}/messages-cursor", response_model=GetAgentMessagesResponse)
def get_agent_messages_cursor(
    agent_id: UUID,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
    before: Optional[UUID] = Query(None, description="Message before which to retrieve the returned messages."),
    limit: int = Query(10, description="Maximum number of messages to retrieve."),

):
    """
    Retrieve the in-context messages of a specific agent. Paginated, provide start and count to iterate.
    """
    actor = server.get_current_user()
    # Validate with the Pydantic model (optional)
    request = GetAgentMessagesCursorRequest(agent_id=agent_id, before=before, limit=limit)

    interface.clear()
    [_, messages] = server.get_agent_recall_cursor(
        user_id=actor._id, agent_id=agent_id, before=request.before, limit=request.limit, reverse=True
    )
    return GetAgentMessagesResponse(messages=messages)

@router.post("/{agent_id}/messages", response_model=UserMessageResponse)
async def send_message(
    # background_tasks: BackgroundTasks,
    agent_id: "UUID",
    server: SyncServer = Depends(get_memgpt_server),
    message: UserMessageRequest = Body(...),
):
    """
    Process a user message and return the agent's response.

    This endpoint accepts a message from a user and processes it through the agent.
    It can optionally stream the response if 'stream' is set to True.
    """
    actor = server.get_current_user()
    return await send_message_to_agent(
        server=server,
        agent_id=agent_id,
        user_id=actor._id,
        role=message.role,
        message=message.message,
        stream_steps=message.stream_steps,
        stream_tokens=message.stream_tokens,
        timestamp=message.timestamp,
        # legacy
        stream_legacy=message.stream,
    )


# TODO: this belongs in a controller!
async def send_message_to_agent(
    agent_id: "UUID",
    role: str,
    message: str,
    stream_legacy: bool,  # legacy
    stream_steps: bool,
    stream_tokens: bool,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
    chat_completion_mode: Optional[bool] = False,
    timestamp: Optional[datetime] = None,
) -> Union[StreamingResponse, UserMessageResponse]:
    """Split off into a separate function so that it can be imported in the /chat/completion proxy."""
    actor = server.get_current_user()
    # handle the legacy mode streaming
    if stream_legacy:
        # NOTE: override
        stream_steps = True
        stream_tokens = False
        include_final_message = False

    if role == "user" or role is None:
        message_func = server.user_message
    elif role == "system":
        message_func = server.system_message
    else:
        raise HTTPException(status_code=500, detail=f"Bad role {role}")

    if not stream_steps and stream_tokens:
        raise HTTPException(status_code=400, detail="stream_steps must be 'true' if stream_tokens is 'true'")

    # For streaming response
    try:

        # Get the generator object off of the agent's streaming interface
        # This will be attached to the POST SSE request used under-the-hood
        memgpt_agent = server._get_or_load_agent(user_id=actor._id, agent_id=agent_id)
        streaming_interface = memgpt_agent.interface
        if not isinstance(streaming_interface, StreamingServerInterface):
            raise ValueError(f"Agent has wrong type of interface: {type(streaming_interface)}")

        # Enable token-streaming within the request if desired
        streaming_interface.streaming_mode = stream_tokens
        # "chatcompletion mode" does some remapping and ignores inner thoughts
        streaming_interface.streaming_chat_completion_mode = chat_completion_mode

        # NOTE: for legacy 'stream' flag
        streaming_interface.nonstreaming_legacy_mode = stream_legacy

        # Offload the synchronous message_func to a separate thread
        streaming_interface.stream_start()
        task = asyncio.create_task(
            asyncio.to_thread(message_func, user_id=user_id, agent_id=agent_id, message=message, timestamp=timestamp)
        )

        if stream_steps:
            # return a stream
            return StreamingResponse(
                sse_async_generator(streaming_interface.get_generator(), finish_message=include_final_message),
                media_type="text/event-stream",
            )
        else:
            # buffer the stream, then return the list
            generated_stream = []
            async for message in streaming_interface.get_generator():
                generated_stream.append(message)
                if "data" in message and message["data"] == "[DONE]":
                    break
            filtered_stream = [d for d in generated_stream if d not in ["[DONE_GEN]", "[DONE_STEP]", "[DONE]"]]
            usage = await task
            return UserMessageResponse(messages=filtered_stream, usage=usage)

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e}")



##### MISSING #######

#@router.post("/{agent_id}/command")
#def run_command(
    #agent_id: "UUID",
    #command: "AgentCommandRequest",
    #interface: "QueuingInterface" = Depends(get_current_interface),
    #server: "SyncServer" = Depends(get_memgpt_server),
#):
    #"""
    #Execute a command on a specified agent.

    #This endpoint receives a command to be executed on an agent. It uses the user and agent identifiers to authenticate and route the command appropriately.

    #Raises an HTTPException for any processing errors.
    #"""
    #actor = server.get_current_user()
    #interface.clear()
    #response = server.run_command(user_id=actor._id,
                                  #agent_id=agent_id,
                                  #command=command.command)

    #return AgentCommandResponse(response=response)

#@router.get("/{agent_id}/config")
#def get_agent_config(
    #agent_id: "UUID",
    #interface: "QueuingInterface" = Depends(get_current_interface),
    #server: "SyncServer" = Depends(get_memgpt_server),
#):
    #"""
    #Retrieve the configuration for a specific agent.

    #This endpoint fetches the configuration details for a given agent, identified by the user and agent IDs.
    #"""
    #actor = server.get_current_user()
    #interface.clear()
    #if not server.ms.get_agent(user_id=actor._id, agent_id=agent_id):
        ## agent does not exist
        #raise HTTPException(status_code=404, detail=f"Agent agent_id={agent_id} not found.")

    #agent_state = server.get_agent_config(user_id=actor._id, agent_id=agent_id)
    ## get sources
    #attached_sources = server.list_attached_sources(agent_id=agent_id)

    ## configs
    #llm_config = LLMConfig(**vars(agent_state.llm_config))
    #embedding_config = EmbeddingConfig(**vars(agent_state.embedding_config))

    #return GetAgentResponse(
        #agent_state=AgentState(
            #id=agent_state.id,
            #name=agent_state.name,
            #user_id=agent_state.user_id,
            #llm_config=llm_config,
            #embedding_config=embedding_config,
            #state=agent_state.state,
            #created_at=int(agent_state.created_at.timestamp()),
            #tools=agent_state.tools,
            #system=agent_state.system,
            #metadata=agent_state._metadata,
        #),
        #last_run_at=None,  # TODO
        #sources=attached_sources,
    #)

#@router.patch("/{agent_id}/rename", response_model=GetAgentResponse)
#def update_agent_name(
    #agent_id: "UUID",
    #agent_rename: AgentRenameRequest,
    #interface: "QueuingInterface" = Depends(get_current_interface),
    #server: "SyncServer" = Depends(get_memgpt_server),
#):
    #"""
    #Updates the name of a specific agent.

    #This changes the name of the agent in the database but does NOT edit the agent's persona.
    #"""
    #valid_name = agent_rename.agent_name
    #actor = server.get_current_user()
    #interface.clear()
    #agent_state = server.rename_agent(user_id=actor._id, agent_id=agent_id, new_agent_name=valid_name)
    ## get sources
    #attached_sources = server.list_attached_sources(agent_id=agent_id)
    #llm_config = LLMConfig(**vars(agent_state.llm_config))
    #embedding_config = EmbeddingConfig(**vars(agent_state.embedding_config))

    #return GetAgentResponse(
        #agent_state=AgentState(
            #id=agent_state.id,
            #name=agent_state.name,
            #user_id=agent_state.user_id,
            #llm_config=llm_config,
            #embedding_config=embedding_config,
            #state=agent_state.state,
            #created_at=int(agent_state.created_at.timestamp()),
            #tools=agent_state.tools,
            #system=agent_state.system,
        #),
        #last_run_at=None,  # TODO
        #sources=attached_sources,
    #)


#@router.get("/{agent_id}/archival/all", response_model=GetAgentArchivalMemoryResponse)
#def get_agent_archival_memory_all(
    #agent_id: "UUID",
    #interface: "QueuingInterface" = Depends(get_current_interface),
    #server: "SyncServer" = Depends(get_memgpt_server),
#):
    #"""
    #Retrieve the memories in an agent's archival memory store (non-paginated, returns all entries at once).
    #"""
    #actor = server.get_current_user()
    #interface.clear()
    #archival_memories = server.get_all_archival_memories(user_id=actor._id, agent_id=agent_id)
    #print("archival_memories:", archival_memories)
    #archival_memory_objects = [ArchivalMemoryObject(id=passage["id"], contents=passage["contents"]) for passage in archival_memories]
    #return GetAgentArchivalMemoryResponse(archival_memory=archival_memory_objects)
