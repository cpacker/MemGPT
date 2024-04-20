# TODO: modify this to run against an actual running server
# def test_list_messages():
#    client = TestClient(app)
#
#    test_user_id = uuid.UUID(MemGPTConfig.load().anon_clientid)
#
#    # create user
#    server = SyncServer()
#    if not server.get_user(test_user_id):
#        print("Creating user in test_list_messages", test_user_id)
#        server.create_user({"id": test_user_id})
#    else:
#        print("User already exists in test_list_messages", test_user_id)
#
#    # write default presets to DB
#    server.initialize_default_presets(test_user_id)
#
#    # test: create agent
#    request_body = {
#        "assistant_name": DEFAULT_PRESET,
#    }
#    print(request_body)
#    response = client.post("/v1/threads", json=request_body)
#    assert response.status_code == 200, f"Error: {response.json()}"
#    agent_id = response.json()["id"]
#    print(response.json())
#
#    # test: insert messages
#    # TODO: eventually implement the "run" functionality
#    request_body = {
#        "content": "Hello, world!",
#        "role": "user",
#    }
#    response = client.post(f"/v1/threads/{str(agent_id)}/messages", json=request_body)
#    assert response.status_code == 200, f"Error: {response.json()}"
#
#    # test: list messages
#    thread_id = str(agent_id)
#    params = {
#        "limit": 10,
#        "order": "desc",
#    }
#    response = client.get(f"/v1/threads/{thread_id}/messages", params=params)
#    assert response.status_code == 200, f"Error: {response.json()}"
#    print(response.json())
#
