# from letta.functions.function_sets.base import send_message


def test_offline_memory_agent():
    pass
    # client = create_client()
    # assert client is not None

    """
    rethink_memory_tool = client.create_tool(rethink_memory)
    send_message_offline_agent_tool = client.create_tool(send_message_offline_agent)
    trigger_rethink_memory_tool = client.create_tool(trigger_rethink_memory)
    conversation_agent = client.create_agent(
        agent_type=AgentType.offline_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[send_message_offline_agent.name, trigger_rethink_memory_tool.name],
        memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        include_base_tools=False,
    )
    assert conversation_agent is not None
    memory_rethink_agent = client.create_agent(
        agent_type=AgentType.offline_memory_agent,
        memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text("offline_memory_persona")),
        tools=[rethink_memory.name],
    )
    new_memory = Block(name="rethink_memory_block", label="memory_rethink_block", value="", limit=2000)
    response = client.user_message(agent_id=conversation_agent.id, message="rethink: Tell me something I don't know about myself.")
    print(response)
    """


if __name__ == "__main__":
    test_offline_memory_agent()
