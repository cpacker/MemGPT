import os
import uuid

from sqlalchemy import MetaData, Table, create_engine

from letta import create_client
from letta.config import LettaConfig
from letta.data_types import AgentState, EmbeddingConfig, LLMConfig
from letta.metadata import MetadataStore
from letta.presets.presets import add_default_tools
from letta.prompts import gpt_system

# Replace this with your actual database connection URL
config = LettaConfig.load()
if config.recall_storage_type == "sqlite":
    DATABASE_URL = "sqlite:///" + os.path.join(config.recall_storage_path, "sqlite.db")
else:
    DATABASE_URL = config.recall_storage_uri
print(DATABASE_URL)
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# defaults
system_prompt = gpt_system.get_system_text("memgpt_chat")

# Reflect the existing table
table = Table("agents", metadata, autoload_with=engine)


# get all agent rows
agent_states = []
with engine.connect() as conn:
    agents = conn.execute(table.select()).fetchall()
    for agent in agents:
        id = uuid.UUID(agent[0])
        user_id = uuid.UUID(agent[1])
        name = agent[2]
        print(f"Migrating agent {name}")
        persona = agent[3]
        human = agent[4]
        system = agent[5]
        preset = agent[6]
        created_at = agent[7]
        llm_config = LLMConfig(**agent[8])
        embedding_config = EmbeddingConfig(**agent[9])
        state = agent[10]
        tools = agent[11]

        state["memory"] = {"human": {"value": human, "limit": 2000}, "persona": {"value": persona, "limit": 2000}}

        agent_state = AgentState(
            id=id,
            user_id=user_id,
            name=name,
            system=system,
            created_at=created_at,
            llm_config=llm_config,
            embedding_config=embedding_config,
            state=state,
            tools=tools,
            _metadata={"human": "migrated", "persona": "migrated"},
        )

        agent_states.append(agent_state)

# remove agents table
agents_model = Table("agents", metadata, autoload_with=engine)
agents_model.drop(engine)

# remove tool table
tool_model = Table("toolmodel", metadata, autoload_with=engine)
tool_model.drop(engine)

# re-create tables and add default tools
ms = MetadataStore(config)
add_default_tools(None, ms)
print("Tools", [tool.name for tool in ms.list_tools()])


for agent in agent_states:
    ms.create_agent(agent)
    print(f"Agent {agent.name} migrated successfully!")

# add another agent to create core memory tool
client = create_client()
dummy_agent = client.create_agent(name="dummy_agent")
tools = client.list_tools()
assert "core_memory_append" in [tool.name for tool in tools]

print("Migration completed successfully!")
