import os

from sqlalchemy import DDL, MetaData, Table, create_engine, update

from memgpt.config import MemGPTConfig
from memgpt.constants import BASE_TOOLS
from memgpt.metadata import MetadataStore
from memgpt.presets.presets import add_default_tools
from memgpt.prompts import gpt_system

# Replace this with your actual database connection URL
config = MemGPTConfig.load()
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

# Using a connection to manage adding columns and committing updates
with engine.connect() as conn:
    trans = conn.begin()
    try:
        # Check and add 'system' column if it does not exist
        if "system" not in table.c:
            ddl_system = DDL("ALTER TABLE agents ADD COLUMN system VARCHAR")
            conn.execute(ddl_system)
            # Reflect the table again to update metadata
            metadata.clear()
            table = Table("agents", metadata, autoload_with=conn)

        # Check and add 'tools' column if it does not exist
        if "tools" not in table.c:
            ddl_tools = DDL("ALTER TABLE agents ADD COLUMN tools JSON")
            conn.execute(ddl_tools)
            # Reflect the table again to update metadata
            metadata.clear()
            table = Table("agents", metadata, autoload_with=conn)

        # Update all existing rows with default values for the new columns
        conn.execute(update(table).values(system=system_prompt, tools=BASE_TOOLS))

        # Commit transaction
        trans.commit()
        print("Columns added and data updated successfully!")

    except Exception as e:
        print("An error occurred:", e)
        trans.rollback()  # Rollback if there are errors

# remove tool table
tool_model = Table("toolmodel", metadata, autoload_with=engine)
tool_model.drop(engine)

# re-create tables and add default tools
ms = MetadataStore(config)
add_default_tools(None, ms)
print("Tools", [tool.name for tool in ms.list_tools()])

print("Migration completed successfully!")
