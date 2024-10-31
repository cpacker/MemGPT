import typer
from swarm import Swarm

from letta import EmbeddingConfig, LLMConfig

"""
This is an example of how to implement the basic example provided by OpenAI for tranferring a conversation between two agents:
https://github.com/openai/swarm/tree/main?tab=readme-ov-file#usage

Before running this example, make sure you have letta>=0.5.0 installed. This example also runs with OpenAI, though you can also change the model by modifying the code:
```bash
export OPENAI_API_KEY=...
pip install letta
````
Then, instead the `examples/swarm` directory, run:
```bash
python simple.py
```
You should see a message output from Agent B.

"""


def transfer_agent_b(self):
    """
    Transfer conversation to agent B.

    Returns:
        str: name of agent to transfer to
    """
    return "agentb"


def transfer_agent_a(self):
    """
    Transfer conversation to agent A.

    Returns:
        str: name of agent to transfer to
    """
    return "agenta"


swarm = Swarm()

# set client configs
swarm.client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))
swarm.client.set_default_llm_config(LLMConfig.default_config(model_name="gpt-4"))

# create tools
transfer_a = swarm.client.create_tool(transfer_agent_a)
transfer_b = swarm.client.create_tool(transfer_agent_b)

# create agents
if swarm.client.get_agent_id("agentb"):
    swarm.client.delete_agent(swarm.client.get_agent_id("agentb"))
if swarm.client.get_agent_id("agenta"):
    swarm.client.delete_agent(swarm.client.get_agent_id("agenta"))
agent_a = swarm.create_agent(name="agentb", tools=[transfer_a.name], instructions="Only speak in haikus")
agent_b = swarm.create_agent(name="agenta", tools=[transfer_b.name])

response = swarm.run(agent_name="agenta", message="Transfer me to agent b by calling the transfer_agent_b tool")
print("Response:")
typer.secho(f"{response}", fg=typer.colors.GREEN)

response = swarm.run(agent_name="agenta", message="My name is actually Sarah. Transfer me to agent b to write a haiku about my name")
print("Response:")
typer.secho(f"{response}", fg=typer.colors.GREEN)

response = swarm.run(agent_name="agenta", message="Transfer me to agent b - I want a haiku with my name in it")
print("Response:")
typer.secho(f"{response}", fg=typer.colors.GREEN)
