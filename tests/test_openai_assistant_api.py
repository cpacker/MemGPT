# from fastapi import FastAPI
# from fastapi.testclient import TestClient
#
# from memgpt.server.rest_api.openai_assistants.assistants import app
#
# def test_list_messages():
#    client = TestClient(app)
#
#    # create user
#
#    # create an agent
#
#    # insert messages
#    # TODO: eventually implement the "run" functionality
#
#    # list messages
#    response = client.get("/v1/threads/123/messages")
#    assert response.status_code == 200
#    assert response.json() == {"messages": []}
#
#    @router.get("/v1/threads/{thread_id}/messages", tags=["assistants"], response_model=ListMessagesResponse)
