# test state saving between client session
# TODO: update this test with correct imports


# def test_save_load(client):
#    """Test that state is being persisted correctly after an /exit
#
#    Create a new agent, and request a message
#
#    Then trigger
#    """
#    assert client is not None, "Run create_agent test first"
#    assert test_agent_state is not None, "Run create_agent test first"
#    assert test_agent_state_post_message is not None, "Run test_user_message test first"
#
#    # Create a new client (not thread safe), and load the same agent
#    # The agent state inside should correspond to the initial state pre-message
#    if os.getenv("OPENAI_API_KEY"):
#        client2 = Letta(quickstart="openai", user_id=test_user_id)
#    else:
#        client2 = Letta(quickstart="letta_hosted", user_id=test_user_id)
#    print(f"\n\n[3] CREATING CLIENT2, LOADING AGENT {test_agent_state.id}!")
#    client2_agent_obj = client2.server._get_or_load_agent(user_id=test_user_id, agent_id=test_agent_state.id)
#    client2_agent_state = client2_agent_obj.update_state()
#    print(f"[3] LOADED AGENT! AGENT {client2_agent_state.id}\n\tmessages={client2_agent_state.state['messages']}")
#
#    # assert test_agent_state == client2_agent_state, f"{vars(test_agent_state)}\n{vars(client2_agent_state)}"
#    def check_state_equivalence(state_1, state_2):
#        """Helper function that checks the equivalence of two AgentState objects"""
#        assert state_1.keys() == state_2.keys(), f"{state_1.keys()}\n{state_2.keys}"
#        for k, v1 in state_1.items():
#            v2 = state_2[k]
#            if isinstance(v1, LLMConfig) or isinstance(v1, EmbeddingConfig):
#                assert vars(v1) == vars(v2), f"{vars(v1)}\n{vars(v2)}"
#            else:
#                assert v1 == v2, f"{v1}\n{v2}"
#
#    check_state_equivalence(vars(test_agent_state), vars(client2_agent_state))
#
#    # Now, write out the save from the original client
#    # This should persist the test message into the agent state
#    client.save()
#
#    if os.getenv("OPENAI_API_KEY"):
#        client3 = Letta(quickstart="openai", user_id=test_user_id)
#    else:
#        client3 = Letta(quickstart="letta_hosted", user_id=test_user_id)
#    client3_agent_obj = client3.server._get_or_load_agent(user_id=test_user_id, agent_id=test_agent_state.id)
#    client3_agent_state = client3_agent_obj.update_state()
#
#    check_state_equivalence(vars(test_agent_state_post_message), vars(client3_agent_state))
#
