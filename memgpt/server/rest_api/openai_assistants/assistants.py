import asyncio
from fastapi import FastAPI
from asyncio import AbstractEventLoop
from enum import Enum
import json
import uuid
from typing import List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, Depends, Body, HTTPException, Query, Path
from pydantic import BaseModel, Field, constr, validator
from starlette.responses import StreamingResponse

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.static_files import mount_static_files
from memgpt.models.openai.messages import OpenAIMessage, Text
from memgpt.data_types import LLMConfig, EmbeddingConfig, Message
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


class ModifyThreadRequest(BaseModel):
    metadata: dict = Field(None, description="Metadata associated with the thread.")


class ModifyMessageRequest(BaseModel):
    metadata: dict = Field(None, description="Metadata associated with the message.")


class ModifyRunRequest(BaseModel):
    metadata: dict = Field(None, description="Metadata associated with the run.")


class CreateMessageRequest(BaseModel):
    role: str = Field(..., description="Role of the message sender (either 'user' or 'system')")
    content: str = Field(..., description="The message content to be processed by the agent.")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs associated with the message.")
    metadata: Optional[dict] = Field(None, description="Metadata associated with the message.")

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


class CreateAssistantFileRequest(BaseModel):
    file_id: str = Field(..., description="The unique identifier of the file.")


class AssistantFile(BaseModel):
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "assistant.file"
    created_at: int = Field(..., description="The unix timestamp of when the file was created.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")


class MessageFile(BaseModel):
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "thread.message.file"
    created_at: int = Field(..., description="The unix timestamp of when the file was created.")
    message_id: str = Field(..., description="The unique identifier of the message.")


@app.post("/v1/assistants/{assistant_id}/files", tags=["assistants"], response_model=AssistantFile)
def create_assistant_file(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    request: CreateAssistantFileRequest = Body(...),
):
    # TODO: add file to assistant
    return AssistantFile(
        id=request.file_id,
        created_at=int(datetime.now().timestamp()),
        assistant_id=assistant_id,
    )


@app.get("/v1/assistants", tags=["assistants"], response_model=List[OpenAIAssistant])
def list_assistants(
    limit: int = Query(1000, description="How many assistants to retrieve."),
    order: str = Query("asc", description="Order of assistants to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: implement list assistants (i.e. list available MemGPT presets)
    pass


@app.get("/v1/assistants/{assistant_id}/files", tags=["assistants"], response_model=List[AssistantFile])
def list_assistant_files(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    limit: int = Query(1000, description="How many files to retrieve."),
    order: str = Query("asc", description="Order of files to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: list attached data sources to preset
    pass


@app.get("/v1/assistants/{assistant_id}", tags=["assistants"], response_model=OpenAIAssistant)
def retrieve_assistant(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
):
    # TODO: get and return preset
    pass


@app.get("/v1/assistants/{assistant_id}/files/{file_id}", tags=["assistants"], response_model=AssistantFile)
def retrieve_assistant_file(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: return data source attached to preset
    pass


@app.post("/v1/assistants/{assistant_id}", tags=["assistants"], response_model=OpenAIAssistant)
def modify_assistant(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    request: CreateAssistantRequest = Body(...),
):
    # TODO: modify preset
    pass


class DeleteAssistantResponse(BaseModel):
    id: str = Field(..., description="The unique identifier of the agent.")
    object: str = "assistant.deleted"
    deleted: bool = Field(..., description="Whether the agent was deleted.")


class DeleteAssistantFileResponse(BaseModel):
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "assistant.file.deleted"
    deleted: bool = Field(..., description="Whether the file was deleted.")


@app.delete("/v1/assistants/{assistant_id}", tags=["assistants"], response_model=DeleteAssistantResponse)
def delete_assistant(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
):
    # TODO: delete preset
    pass


@app.delete("/v1/assistants/{assistant_id}/files/{file_id}", tags=["assistants"], response_model=DeleteAssistantFileResponse)
def delete_assistant_file(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: delete source on preset
    pass


@app.post("/v1/threads/", tags=["assistants"], response_model=OpenAIThread)
def create_thread(request: CreateThreadRequest = Body(...)):

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


@app.get("/v1/threads/{thread_id}", tags=["assistants"], response_model=OpenAIThread)
def retrieve_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
):
    agent = server.get_agent(uuid.UUID(thread_id))
    return OpenAIThread(
        id=str(agent.id),
        created_at=int(agent.created_at.timestamp()),
    )


@app.get("/v1/threads/{thread_id}", tags=["assistants"], response_model=OpenAIThread)
def modify_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: ModifyThreadRequest = Body(...),
):
    # TODO: add agent metadata so this can be modified
    pass


class DeleteThreadResponse(BaseModel):
    id: str = Field(..., description="The unique identifier of the agent.")
    object: str = "thread.deleted"
    deleted: bool = Field(..., description="Whether the agent was deleted.")


@app.delete("/v1/threads/{thread_id}", tags=["assistants"], response_model=DeleteThreadResponse)
def delete_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
):
    # TODO: delete agent
    pass


@app.post("/v1/threads/{thread_id}/messages", tags=["assistants"], response_model=OpenAIMessage)
def create_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateMessageRequest = Body(...),
):
    user_id = uuid.UUID(request.user_id)
    agent_id = uuid.UUID(thread_id)
    # TODO: need to add a buffer/queue to server and pull on .step()
    message = Message(
        user_id=user_id,
        agent_id=agent_id,
        role=request.role,
        text=request.content,
    )
    server.user_message(
        user_id=user_id,
        agent_id=agent_id,
        message=message,
    )
    return OpenAIMessage(
        id=str(message.id),
        created_at=int(message.created_at.timestamp()),
        content=[Text(text=message.text)],
        role=message.role,
        thread_id=str(message.agent_id),
        assistant_id=DEFAULT_PRESET,  # TODO: update this
        # file_ids=message.file_ids,
        # metadata=message.metadata,
    )


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


app.get("/v1/threads/{thread_id}/messages/{message_id}", tags=["assistants"], response_model=OpenAIMessage)


def retrieve_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
):
    message_id = uuid.UUID(message_id)
    agent_id = uuid.UUID(thread_id)
    message = server.get_agent_message(agent_id, message_id)
    return OpenAIMessage(
        id=str(message.id),
        created_at=int(message.created_at.timestamp()),
        content=[Text(text=message.text)],
        role=message.role,
        thread_id=str(message.agent_id),
        assistant_id=DEFAULT_PRESET,  # TODO: update this
        # file_ids=message.file_ids,
        # metadata=message.metadata,
    )


@app.get("/v1/threads/{thread_id}/messages/{message_id}/files/{file_id}", tags=["assistants"], response_model=MessageFile)
def retrieve_message_file(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: implement?
    pass


class Function(BaseModel):
    name: str = Field(..., description="The name of the function.")
    arguments: str = Field(..., description="The arguments of the function.")


class ToolCall(BaseModel):
    id: str = Field(..., description="The unique identifier of the tool call.")
    type: str = "function"
    function: Function = Field(..., description="The function call.")


class RequiredAction(BaseModel):
    type: str = "submit_tool_outputs"
    submit_tool_outputs: List[ToolCall]


class OpenAIError(BaseModel):
    code: str = Field(..., description="The error code.")
    message: str = Field(..., description="The error message.")


class OpenAIUsage(BaseModel):
    completion_tokens: int = Field(..., description="The number of tokens used for the run.")
    prompt_tokens: int = Field(..., description="The number of tokens used for the prompt.")
    total_tokens: int = Field(..., description="The total number of tokens used for the run.")


class OpenAIMessageCreationStep(BaseModel):
    type: str = "message_creation"
    message_id: str = Field(..., description="The unique identifier of the message.")


class OpenAIToolCallsStep(BaseModel):
    type: str = "tool_calls"
    tool_calls: List[ToolCall] = Field(..., description="The tool calls.")


class OpenAIRun(BaseModel):
    id: str = Field(..., description="The unique identifier of the run.")
    object: str = "thread.run"
    created_at: int = Field(..., description="The unix timestamp of when the run was created.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    status: str = Field(..., description="The status of the run.")
    required_action: Optional[RequiredAction] = Field(None, description="The required action of the run.")
    last_error: Optional[OpenAIError] = Field(None, description="The last error of the run.")
    expires_at: int = Field(..., description="The unix timestamp of when the run expires.")
    started_at: Optional[int] = Field(None, description="The unix timestamp of when the run started.")
    cancelled_at: Optional[int] = Field(None, description="The unix timestamp of when the run was cancelled.")
    failed_at: Optional[int] = Field(None, description="The unix timestamp of when the run failed.")
    completed_at: Optional[int] = Field(None, description="The unix timestamp of when the run completed.")
    model: str = Field(..., description="The model used by the run.")
    instructions: str = Field(..., description="The instructions for the run.")
    tools: List[ToolCall] = Field(..., description="The tools used by the run.")  # TODO: also add code interpreter / retrieval
    file_ids: List[str] = Field(..., description="List of file IDs associated with the run.")
    metadata: dict = Field(..., description="Metadata associated with the run.")
    usage: Optional[OpenAIUsage] = Field(None, description="The usage of the run.")


class OpenAIRunStep(BaseModel):
    id: str = Field(..., description="The unique identifier of the run step.")
    object: str = "thread.run.step"
    created_at: int = Field(..., description="The unix timestamp of when the run step was created.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    run_id: str = Field(..., description="The unique identifier of the run.")
    type: str = Field(..., description="The type of the run step.")  # message_creation, tool_calls
    status: str = Field(..., description="The status of the run step.")
    step_defaults: Union[OpenAIToolCallsStep, OpenAIMessageCreationStep] = Field(..., description="The step defaults.")
    last_error: Optional[OpenAIError] = Field(None, description="The last error of the run step.")
    expired_at: Optional[int] = Field(None, description="The unix timestamp of when the run step expired.")
    failed_at: Optional[int] = Field(None, description="The unix timestamp of when the run failed.")
    completed_at: Optional[int] = Field(None, description="The unix timestamp of when the run completed.")
    usage: Optional[OpenAIUsage] = Field(None, description="The usage of the run.")


@app.post("/v1/threads/{thread_id}/messages/{message_id}", tags=["assistants"], response_model=OpenAIMessage)
def modify_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    request: ModifyMessageRequest = Body(...),
):
    # TODO: add metada field to message so this can be modified
    pass


class CreateRunRequest(BaseModel):
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    model: str = Field(..., description="The model used by the run.")
    instructions: str = Field(..., description="The instructions for the run.")
    additional_instructions: Optional[str] = Field(None, description="Additional instructions for the run.")
    tools: Optional[List[ToolCall]] = Field(None, description="The tools used by the run (overrides assistant).")
    metadata: Optional[dict] = Field(None, description="Metadata associated with the run.")


class CreateThreadRunRequest(BaseModel):
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    thread: OpenAIThread = Field(..., description="The thread to run.")
    model: str = Field(..., description="The model used by the run.")
    instructions: str = Field(..., description="The instructions for the run.")
    tools: Optional[List[ToolCall]] = Field(None, description="The tools used by the run (overrides assistant).")
    metadata: Optional[dict] = Field(None, description="Metadata associated with the run.")


@app.post("/v1/threads/{thread_id}/runs", tags=["assistants"], response_model=OpenAIMessage)
def create_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateRunRequest = Body(...),
):
    # TODO: need to implement lazy process of messages, then can use this to execute run
    pass


@app.post("/v1/threads/runs", tags=["assistants"], response_model=OpenAIRun)
def create_thread_and_run(
    request: CreateThreadRunRequest = Body(...),
):
    # TODO: add a bunch of messages and execute
    pass


@app.get("/v1/threads/{thread_id}/runs", tags=["assistants"], response_model=List[OpenAIRun])
def list_runs(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    limit: int = Query(1000, description="How many runs to retrieve."),
    order: str = Query("asc", description="Order of runs to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: store run information in a DB so it can be returned here
    pass


@app.get("/v1/threads/{thread_id}/runs/{run_id}/steps", tags=["assistants"], response_model=List[OpenAIRunStep])
def list_run_steps(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    limit: int = Query(1000, description="How many run steps to retrieve."),
    order: str = Query("asc", description="Order of run steps to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: store run information in a DB so it can be returned here
    pass


@app.get("/v1/threads/{thread_id}/runs/{run_id}", tags=["assistants"], response_model=OpenAIRun)
def retrieve_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
):
    pass


@app.get("/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}", tags=["assistants"], response_model=OpenAIRunStep)
def retrieve_run_step(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    step_id: str = Path(..., description="The unique identifier of the run step."),
):
    pass


@app.post("/v1/threads/{thread_id}/runs/{run_id}", tags=["assistants"], response_model=OpenAIRun)
def modify_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    request: ModifyRunRequest = Body(...),
):
    pass


class ToolCallOutput(BaseModel):
    tool_call_id: str = Field(..., description="The unique identifier of the tool call.")
    output: str = Field(..., description="The output of the tool call.")


class SubmitToolOutputsToRunRequest(BaseModel):
    tools_outputs: List[ToolCallOutput] = Field(..., description="The tool outputs to submit.")


@app.post("/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs", tags=["assistants"], response_model=OpenAIRun)
def submit_tool_outputs_to_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    request: SubmitToolOutputsToRunRequest = Body(...),
):
    pass


@app.post("/v1/threads/{thread_id}/runs/{run_id}/cancel", tags=["assistants"], response_model=OpenAIRun)
def cancel_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
):
    pass
