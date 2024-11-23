from letta import EmbeddingConfig, LLMConfig, create_client
from letta.schemas.tool_rule import TerminalToolRule

client = create_client()
# set automatic defaults for LLM/embedding config
client.set_default_llm_config(LLMConfig.default_config(model_name="gpt-4"))
client.set_default_embedding_config(EmbeddingConfig.default_config(model_name="text-embedding-ada-002"))


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


# create a tool from the function
tool = client.create_tool(roll_d20)
print(f"Created tool with name {tool.name}")

# create a new agent
agent_state = client.create_agent(
    # create the agent with an additional tool
    tools=[tool.name],
    # add tool rules that terminate execution after specific tools
    tool_rules=[
        # exit after roll_d20 is called
        TerminalToolRule(tool_name=tool.name),
        # exit after send_message is called (default behavior)
        TerminalToolRule(tool_name="send_message"),
    ],
)
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
