@router.post("/threads", tags=["threads"], response_model=OpenAIThread)
def create_thread(request: CreateThreadRequest = Body(...)):
    # TODO: use requests.description and requests.metadata fields
    # TODO: handle requests.file_ids and requests.tools
    # TODO: eventually allow request to override embedding/llm model

    print("Create thread/agent", request)
    # create a memgpt agent
    agent_state = server.create_agent(
        user_id=user_id,
    )
    # TODO: insert messages into recall memory
    return OpenAIThread(
        id=str(agent_state.id),
        created_at=int(agent_state.created_at.timestamp()),
    )


@router.get("/threads/{thread_id}", tags=["threads"], response_model=OpenAIThread)
def retrieve_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
):
    agent = server.get_agent(uuid.UUID(thread_id))
    return OpenAIThread(
        id=str(agent.id),
        created_at=int(agent.created_at.timestamp()),
    )


@router.get("/threads/{thread_id}", tags=["threads"], response_model=OpenAIThread)
def modify_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: ModifyThreadRequest = Body(...),
):
    # TODO: add agent metadata so this can be modified
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.delete("/threads/{thread_id}", tags=["threads"], response_model=DeleteThreadResponse)
def delete_thread(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
):
    # TODO: delete agent
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/threads/{thread_id}/messages", tags=["messages"], response_model=OpenAIMessage)
def create_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateMessageRequest = Body(...),
):
    agent_id = uuid.UUID(thread_id)
    # create message object
    message = Message(
        user_id=user_id,
        agent_id=agent_id,
        role=request.role,
        text=request.content,
    )
    agent = server._get_or_load_agent(user_id=user_id, agent_id=agent_id)
    # add message to agent
    agent._append_to_messages([message])

    openai_message = OpenAIMessage(
        id=str(message.id),
        created_at=int(message.created_at.timestamp()),
        content=[Text(text=message.text)],
        role=message.role,
        thread_id=str(message.agent_id),
        assistant_id=settings.preset,  # TODO: update this
        # file_ids=message.file_ids,
        # metadata=message.metadata,
    )
    return openai_message


@router.get("/threads/{thread_id}/messages", tags=["messages"], response_model=ListMessagesResponse)
def list_messages(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    limit: int = Query(1000, description="How many messages to retrieve."),
    order: str = Query("asc", description="Order of messages to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    after_uuid = uuid.UUID(after) if before else None
    before_uuid = uuid.UUID(before) if before else None
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
            assistant_id=settings.preset,  # TODO: update this
            # file_ids=message.file_ids,
            # metadata=message.metadata,
        )
        for message in json_messages
    ]
    print("MESSAGES", openai_messages)
    # TODO: cast back to message objects
    return ListMessagesResponse(messages=openai_messages)


router.get("/threads/{thread_id}/messages/{message_id}", tags=["messages"], response_model=OpenAIMessage)


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
        assistant_id=settings.preset,  # TODO: update this
        # file_ids=message.file_ids,
        # metadata=message.metadata,
    )


@router.get("/threads/{thread_id}/messages/{message_id}/files/{file_id}", tags=["messages"], response_model=MessageFile)
def retrieve_message_file(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: implement?
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/threads/{thread_id}/messages/{message_id}", tags=["messages"], response_model=OpenAIMessage)
def modify_message(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    message_id: str = Path(..., description="The unique identifier of the message."),
    request: ModifyMessageRequest = Body(...),
):
    # TODO: add metada field to message so this can be modified
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/threads/{thread_id}/runs", tags=["runs"], response_model=OpenAIRun)
def create_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    request: CreateRunRequest = Body(...),
):
    # TODO: add request.instructions as a message?
    agent_id = uuid.UUID(thread_id)
    # TODO: override preset of agent with request.assistant_id
    agent = server._get_or_load_agent(user_id=user_id, agent_id=agent_id)
    agent.step(user_message=None)  # already has messages added
    run_id = str(uuid.uuid4())
    create_time = int(get_utc_time().timestamp())
    return OpenAIRun(
        id=run_id,
        created_at=create_time,
        thread_id=str(agent_id),
        assistant_id=settings.preset,  # TODO: update this
        status="completed",  # TODO: eventaully allow offline execution
        expires_at=create_time,
        model=agent.agent_state.llm_config.model,
        instructions=request.instructions,
    )


@router.post("/threads/runs", tags=["runs"], response_model=OpenAIRun)
def create_thread_and_run(
    request: CreateThreadRunRequest = Body(...),
):
    # TODO: add a bunch of messages and execute
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/threads/{thread_id}/runs", tags=["runs"], response_model=List[OpenAIRun])
def list_runs(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    limit: int = Query(1000, description="How many runs to retrieve."),
    order: str = Query("asc", description="Order of runs to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: store run information in a DB so it can be returned here
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/threads/{thread_id}/runs/{run_id}/steps", tags=["runs"], response_model=List[OpenAIRunStep])
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


@router.get("/threads/{thread_id}/runs/{run_id}", tags=["runs"], response_model=OpenAIRun)
def retrieve_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/threads/{thread_id}/runs/{run_id}/steps/{step_id}", tags=["runs"], response_model=OpenAIRunStep)
def retrieve_run_step(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    step_id: str = Path(..., description="The unique identifier of the run step."),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/threads/{thread_id}/runs/{run_id}", tags=["runs"], response_model=OpenAIRun)
def modify_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    request: ModifyRunRequest = Body(...),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/threads/{thread_id}/runs/{run_id}/submit_tool_outputs", tags=["runs"], response_model=OpenAIRun)
def submit_tool_outputs_to_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
    request: SubmitToolOutputsToRunRequest = Body(...),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/threads/{thread_id}/runs/{run_id}/cancel", tags=["runs"], response_model=OpenAIRun)
def cancel_run(
    thread_id: str = Path(..., description="The unique identifier of the thread."),
    run_id: str = Path(..., description="The unique identifier of the run."),
):
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")
