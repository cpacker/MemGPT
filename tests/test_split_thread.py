from memgpt.client.client import create_client


def test_split_thread_creation():
    client = create_client()
    assert client is not None

    agent_state = client.create_agent()
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="Hello")
    assert response is not None

    for i in response.messages:
        print(i.id)
        print(i.text)
        print("\n" * 2)


if __name__ == "__main__":
    test_split_thread_creation()
