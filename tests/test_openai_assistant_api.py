from fastapi import FastAPI
from fastapi.testclient import TestClient
import uuid

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.server import app
from memgpt.constants import DEFAULT_PRESET


def test_list_messages():
    client = TestClient(app)

    test_user_id = uuid.uuid4()

    # create user
    server = SyncServer()
    server.create_user({"id": test_user_id})

    # test: create agent
    request_body = {
        "user_id": str(test_user_id),
        "assistant_name": DEFAULT_PRESET,
    }
    print(request_body)
    response = client.post("/v1/threads", json=request_body)
    assert response.status_code == 200, f"Error: {response.json()}"
    agent_id = response.json()["id"]
    print(response.json())

    # test: insert messages
    # TODO: eventually implement the "run" functionality
    request_body = {
        "user_id": str(test_user_id),
        "content": "Hello, world!",
        "role": "user",
    }
    response = client.post(f"/v1/threads/{str(agent_id)}/messages", json=request_body)
    assert response.status_code == 200, f"Error: {response.json()}"

    # test: list messages
    thread_id = str(agent_id)
    params = {
        "limit": 10,
        "order": "desc",
        # "after": "",
        "user_id": str(test_user_id),
    }
    response = client.get(f"/v1/threads/{thread_id}/messages", params=params)
    assert response.status_code == 200, f"Error: {response.json()}"
    print(response.json())
