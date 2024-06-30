from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine

from memgpt.config import MemGPTConfig
from memgpt.constants import BASE_TOOLS
from memgpt.metadata import MetadataStore
from memgpt.presets.presets import add_default_tools
from memgpt.prompts import gpt_system

# Replace this with your actual database connection URL
DATABASE_URL = "sqlite:////Users/sarahwooders/.memgpt/sqlite.db"

engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Reflect the existing table
agent_table = Table("agents", metadata, autoload_with=engine)
system_prompt = gpt_system.get_system_text("memgpt_chat")  # fill in default system prompt
tools = BASE_TOOLS


# Add new columns if they don't exist
if "system" not in agent_table.c:
    new_column_system = Column("system", String)
    new_column_system.create(table)
    stmt = update(table).values(system=system_prompt)
    engine.execute(stmt)

if "tools" not in agent_table.c:
    new_column_tools = Column("tools", JSON)
    new_column_tools.create(table)
    stmt = update(table).values(system=tools)
    engine.execute(stmt)

# remove tool table
tool_model = Table("toolmodel", metadata, autoload_with=engine)
tool_model.drop(engine)

# re-create tables and add default tools
config = MemGPTConfig.load()
ms = MetadataStore(config)
add_default_tools(None, ms)
print("Tools", [tool.name for tool in ms.list_tools()])

print("Migration completed successfully!")
