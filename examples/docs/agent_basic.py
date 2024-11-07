from letta import EmbeddingConfig, LLMConfig, create_client

client = create_client()

# set automatic defaults for LLM/embedding config
client.set_default_llm_config(LLMConfig.default_config(model_name="gpt-4"))
client.set_default_embedding_config(EmbeddingConfig.default_config(model_name="text-embedding-ada-002"))

# create a new agent
agent_state = client.create_agent()
print(f"Created agent with name {agent_state.name} and unique ID {agent_state.id}")

# Message an agent
response = client.send_message(agent_id=agent_state.id, role="user", message="hello")
print("Usage", response.usage)
print("Agent messages", response.messages)

# list all agents
agents = client.list_agents()

# get the agent by ID
agent_state = client.get_agent(agent_id=agent_state.id)

# get the agent by name
agent_id = client.get_agent_id(agent_name=agent_state.name)
agent_state = client.get_agent(agent_id=agent_id)

# delete an agent
client.delete_agent(agent_id=agent_state.id)
