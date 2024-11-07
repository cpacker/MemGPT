from letta import EmbeddingConfig, LLMConfig, create_client

client = create_client()
# set automatic defaults for LLM/embedding config
client.set_default_llm_config(
    LLMConfig(model="gpt-4", model_endpoint_type="openai", model_endpoint="https://api.openai.com/v1", context_window=8000)
)
client.set_default_embedding_config(
    EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536,
        embedding_chunk_size=300,
    )
)


# define a function with a docstring
def roll_d20() -> str:
    """
    Simulate the roll of a 20-sided die (d20).

    This function generates a random integer between 1 and 20, inclusive,
    which represents the outcome of a single roll of a d20.

    Returns:
        int: A random integer between 1 and 20, representing the die roll.

    Example:
        >>> roll_d20()
        15  # This is an example output and may vary each time the function is called.
    """
    import random

    dice_role_outcome = random.randint(1, 20)
    output_string = f"You rolled a {dice_role_outcome}"
    return output_string


tool = client.create_tool(roll_d20, name="roll_dice")

# create a new agent
agent_state = client.create_agent(tools=[tool.name])
print(f"Created agent with name {agent_state.name} with tools {agent_state.tools}")

# Message an agent
response = client.send_message(agent_id=agent_state.id, role="user", message="roll a dice")
print("Usage", response.usage)
print("Agent messages", response.messages)

# remove a tool from the agent
client.remove_tool_from_agent(agent_id=agent_state.id, tool_id=tool.id)

# add a tool to the agent
client.add_tool_to_agent(agent_id=agent_state.id, tool_id=tool.id)

client.delete_agent(agent_id=agent_state.id)

# create an agent with only a subset of default tools
agent_state = client.create_agent(include_base_tools=False, tools=[tool.name, "send_message"])

# message the agent to search archival memory (will be unable to do so)
response = client.send_message(agent_id=agent_state.id, role="user", message="search your archival memory")
print("Usage", response.usage)
print("Agent messages", response.messages)

client.delete_agent(agent_id=agent_state.id)
