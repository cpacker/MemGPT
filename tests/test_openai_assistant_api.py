from fastapi import FastAPI
from fastapi.testclient import TestClient
import uuid

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.openai_assistants.assistants import app
from memgpt.constants import DEFAULT_PRESET


def test_list_messages():
    client = TestClient(app)

    test_user_id = uuid.uuid4()

    # create user
    server = SyncServer()
    server.create_user({"id": test_user_id})

    # POST request to create agent
    request_body = {
        "user_id": str(test_user_id),
        "assistant_name": DEFAULT_PRESET,
    }
    print(request_body)
    response = client.post("/v1/threads", json=request_body)
    assert response.status_code == 200, f"Error: {response.json()}"
    agent_id = response.json()["id"]
    print(response.json())

    # insert messages
    # TODO: eventually implement the "run" functionality

    # list messages
    thread_id = str(agent_id)
    response = client.get(f"/v1/threads/{thread_id}/messages")
    assert response.status_code == 200
    print(response.json())
