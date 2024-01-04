import os
import json

from tqdm import tqdm

from memgpt.constants import MEMGPT_DIR
from memgpt.utils import version_less_than

# This is the version where the breaking change was made
VERSION_CUTOFF = "0.2.12"
MIGRATION_FILE_NAME = f"{VERSION_CUTOFF}_migration.json"


def agent_is_migrateable(agent_name: str):
    """Determine whether or not the agent folder is a migration target"""
    agent_folder = os.path.join(MEMGPT_DIR, "agents", agent_name)

    if not os.path.exists(agent_folder):
        raise ValueError(f"Folder {agent_folder} does not exist")

    agent_config_file = os.path.join(agent_folder, "config.json")
    if not os.path.exists(agent_config_file):
        raise ValueError(f"Agent folder {agent_folder} does not have a config file")

    try:
        with open(agent_config_file, "r") as fh:
            agent_config = json.load(fh)
    except Exception as e:
        raise ValueError(f"Failed to load agent config file ({agent_config_file}), error = {e}")

    if "memgpt_version" not in vars(agent_config) or version_less_than(agent_config.memgpt_version, VERSION_CUTOFF):
        return True
    else:
        return False


def write_old_agent_to_migration_json(agent_name: str, overwrite: bool = False) -> str:
    """Take an old (pre 0.2.11) agent folder, and write out the contents to a JSON directory

    Contents written to the JSON include:
    - agent properties (system, functions, core memory, in-context messages)
    - persistence manager properties (recall memory, archival memory)
    """
    agent_folder = os.path.join(MEMGPT_DIR, "agents", agent_name)
    migration_file = os.path.join(agent_folder, MIGRATION_FILE_NAME)

    if os.path.exists(migration_file):
        if not overwrite:
            raise ValueError(f"Migration file ({migration_file}) already exists")
        else:
            pass

    # TODO


def create_new_agent_from_migration_json(migration_json_file: str) -> str:
    """Create a new agent with the same name as the old agent, and programmatically load data in"""
    # TODO
    pass


def migrate_agent(agent_name: str):
    """Write out old agents to intermediate migration JSON, then use the JSON to instantiate a new agent"""
    try:
        if not agent_is_migrateable(agent_name=agent_name):
            return
    except ValueError as e:
        return

    try:
        migration_file = write_old_agent_to_migration_json(agent_name=agent_name, overwrite=False)
    except ValueError as e:
        # The migration file may have already been created
        return

    try:
        new_agent = create_new_agent_from_migration_json(migration_json_file=migration_file)
    except ValueError as e:
        # The agent name may already exist
        return

    print(f"Successfully migrated agent {new_agent}")


def migrate_all_agents():
    """Scan over all agent folders in MEMGPT_DIR and migrate each agent."""
    agents_dir = os.path.join(MEMGPT_DIR, "agents")

    # Ensure the directory exists
    if not os.path.exists(agents_dir):
        raise ValueError(f"Directory {agents_dir} does not exist.")

    # Get a list of all folders in agents_dir
    agent_folders = [f for f in os.listdir(agents_dir) if os.path.isdir(os.path.join(agents_dir, f))]

    # Iterate over each folder with a tqdm progress bar
    count = 0
    candidates = 0
    for agent_name in tqdm(agent_folders, desc="Migrating agents"):
        # Assuming migrate_agent is a function that takes the agent name and performs migration
        try:
            if agent_is_migrateable(agent_name=agent_name):
                candidates += 1
                migrate_agent(agent_name)
                count += 1
            else:
                continue
        except Exception as e:
            print(f"Migrating {agent_name} failed with: {e}")

    print(f"Successfully migrated {count}/{candidates} migration targets ({len(agent_folders)} agents total)")
