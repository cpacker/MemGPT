import uuid
from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Query

from letta.constants import DEFAULT_PRESET
from letta.schemas.agent import CreateAgent
from letta.schemas.enums import MessageRole
from letta.schemas.message import Message
from letta.schemas.openai.openai import (
    MessageFile,
    OpenAIMessage,
    OpenAIRun,
    OpenAIRunStep,
    OpenAIThread,
    Text,
)
from letta.server.rest_api.routers.openai.assistants.schemas import (
    CreateMessageRequest,
    CreateRunRequest,
    CreateThreadRequest,
    CreateThreadRunRequest,
    DeleteThreadResponse,
    ListMessagesResponse,
    ModifyMessageRequest,
    ModifyRunRequest,
    ModifyThreadRequest,
    OpenAIThread,
    SubmitToolOutputsToRunRequest,
)
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

if TYPE_CHECKING:
    from letta.utils import get_utc_time


# TODO: implement mechanism for creating/authenticating users associated with a bearer token
router = APIRouter(prefix="/v1/threads", tags=["threads"])


@router.post("/", response_model=OpenAIThread)
def create_thread(
    request: CreateThreadRequest = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    # TODO: use requests.description and requests.metadata fields
    # TODO: handle requests.file_ids and requests.tools
    # TODO: eventually allow request to override embedding/llm model
    actor = server.get_user_or_default(user_id=user_id)

    print("Create thread/agent", request)
    # create a letta agent
    agent_state = server.create_agent(
        request=CreateAgent(),
        user_id=actor.id,
    )
    # TODO: insert messages into recall memory
    return OpenAIThread(
        id=str(agent_state.id),
        created_at=int(agent_state.created_at.timestamp()),
        metadata={},  # TODO add metadata?
    )


@router.get("/{thread_id}", response_model=OpenAIThread)
def retrieve_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id=user_id)
    agent = server.get_agent(user_id=actor.id, agent_id=thread_id)
    assert agent is not None
    return OpenAIThread(
        id=str(agent.id),
        created_at=int(agent.created_at.timestamp()),
        metadata={},  # TODO add metadata?
    )


@router.get("/{thread_id}", response_model=OpenAIThread)
def modify_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: ModifyThreadRequest = Body(...),
):
    # TODO: add agent metadata so this can be modified
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.delete("/{thread_id}", response_model=DeleteThreadResponse)
def delete_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
):
    # TODO: delete agent
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{thread_id}/messages", tags=["messages"], response_model=OpenAIMessage)
def create_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateMessageRequest = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id=user_id)
    agent_id = thread_id
    # create message object
    message = Message(
        user_id=actor.id,
        agent_id=agent_id,
        role=MessageRole(request.role),
        text=request.content,
        model=None,
        tool_calls=None,
        tool_call_id=None,
        name=None,
    )
    agent = server._get_or_load_agent(agent_id=agent_id)
    # add message to agent
    agent._append_to_messages([message])

    openai_message = OpenAIMessage(
        id=str(message.id),
        created_at=int(message.created_at.timestamp()),
        content=[Text(text=(message.text if message.text else ""))],
        role=message.role,
        thread_id=str(message.agent_id),
        assistant_id=DEFAULT_PRESET,  # TODO: update this
        # TODO(sarah) fill in?
        run_id=None,
        file_ids=None,
        metadata=None,
        # file_ids=message.file_ids,
        # metadata=message.metadata,
    )
    return openai_message


@router.get("/{thread_id}/messages", tags=["messages"], response_model=ListMessagesResponse)
def list_messages(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    limit: int = Query(1000, description="How many messages to retrieve."),
    order: str = Query("asc", description="Order of messages to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id)
    after_uuid = after if before else None
    before_uuid = before if before else None
    agent_id = thread_id
    reverse = True if (order == "desc") else False
    json_messages = server.get_agent_recall_cursor(
        user_id=actor.id,
        agent_id=agent_id,
        limit=limit,
        after=after_uuid,
        before=before_uuid,
        order_by="created_at",
        reverse=reverse,
        return_message_object=True,
    )
    assert isinstance(json_messages, List)
    assert all([isinstance(message, Message) for message in json_messages])
    assert isinstance(json_messages[0], Message)
    print(json_messages[0].text)
    # convert to openai style messages
    openai_messages = []
    for message in json_messages:
        assert isinstance(message, Message)
        openai_messages.append(
            OpenAIMessage(
                id=str(message.id),
                created_at=int(message.created_at.timestamp()),
                content=[Text(text=(message.text if message.text else ""))],
                role=str(message.role),
                thread_id=str(message.agent_id),
                assistant_id=DEFAULT_PRESET,  # TODO: update this
                # TODO(sarah) fill in?
                run_id=None,
                file_ids=None,
                metadata=None,
                # file_ids=message.file_ids,
                # metadata=message.metadata,
            )
        )
    print("MESSAGES", openai_messages)
    # TODO: cast back to message objects
    return ListMessagesResponse(messages=openai_messages)


@router.get("/{thread_id}/messages/{message_id}", tags=["messages"], response_model=OpenAIMessage)
def retrieve_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    server: SyncServer = Depends(get_letta_server),
):
    agent_id = thread_id
    message = server.get_agent_message(agent_id=agent_id, message_id=message_id)
    assert message is not None
    return OpenAIMessage(
        id=message_id,
        created_at=int(message.created_at.timestamp()),
        content=[Text(text=(message.text if message.text else ""))],
        role=message.role,
        thread_id=str(message.agent_id),
        assistant_id=DEFAULT_PRESET,  # TODO: update this
        # TODO(sarah) fill in?
        run_id=None,
        file_ids=None,
        metadata=None,
        # file_ids=message.file_ids,
        # metadata=message.metadata,
    )


@router.get("/{thread_id}/messages/{message_id}/files/{file_id}", tags=["messages"], response_model=MessageFile)
def retrieve_message_file(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: implement?
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{thread_id}/messages/{message_id}", tags=["messages"], response_model=OpenAIMessage)
def modify_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    request: ModifyMessageRequest = Body(...),
):
    # TODO: add metada field to message so this can be modified
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{thread_id}/runs", tags=["runs"], response_model=OpenAIRun)
def create_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateRunRequest = Body(...),
    server: SyncServer = Depends(get_letta_server),
):

    # TODO: add request.instructions as a message?
    agent_id = thread_id
    # TODO: override preset of agent with request.assistant_id
    agent = server._get_or_load_agent(agent_id=agent_id)
    agent.inner_step(messages=[])  # already has messages added
    run_id = str(uuid.uuid4())
    create_time = int(get_utc_time().timestamp())
    return OpenAIRun(
        id=run_id,
        created_at=create_time,
        thread_id=str(agent_id),
        assistant_id=DEFAULT_PRESET,  # TODO: update this
        status="completed",  # TODO: eventaully allow offline execution
        expires_at=create_time,
        model=agent.agent_state.llm_config.model,
        instructions=request.instructions,
    )


@router.post("/runs", tags=["runs"], response_model=OpenAIRun)
def create_thread_and_run(
    request: CreateThreadRunRequest = Body(...),
):
    # TODO: add a bunch of messages and execute
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{thread_id}/runs", tags=["runs"], response_model=List[OpenAIRun])
def list_runs(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    limit: int = Query(1000, description="How many runs to retrieve."),
    order: str = Query("asc", description="Order of runs to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: store run information in a DB so it can be returned here
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{thread_id}/runs/{run_id}/steps", tags=["runs"], response_model=List[OpenAIRunStep])
def list_run_steps(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    limit: int = Query(1000, description="How many run steps to retrieve."),
    order: str = Query("asc", description="Order of run steps to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: store run information in a DB so it can be returned here
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{thread_id}/runs/{run_id}", tags=["runs"], response_model=OpenAIRun)
def retrieve_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{thread_id}/runs/{run_id}/steps/{step_id}", tags=["runs"], response_model=OpenAIRunStep)
def retrieve_run_step(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    step_id: str = Path(..., description="The unique identifier of the run step."),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{thread_id}/runs/{run_id}", tags=["runs"], response_model=OpenAIRun)
def modify_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    request: ModifyRunRequest = Body(...),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{thread_id}/runs/{run_id}/submit_tool_outputs", tags=["runs"], response_model=OpenAIRun)
def submit_tool_outputs_to_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    request: SubmitToolOutputsToRunRequest = Body(...),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{thread_id}/runs/{run_id}/cancel", tags=["runs"], response_model=OpenAIRun)
def cancel_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")
