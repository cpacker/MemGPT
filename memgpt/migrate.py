import os
import json

from tqdm import tqdm

from memgpt.utils import MEMGPT_DIR, version_less_than

# This is the version where the breaking change was made
VERSION_CUTOFF = "0.2.12"


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

    if not hasattr(agent_config, "memgpt_version") or version_less_than(agent_config.memgpt_version, VERSION_CUTOFF):
        return True
    else:
        return False


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
                # migrate_agent(agent_name)
                print("fake migrate")
                count += 1
            else:
                continue
        except Exception as e:
            print(f"Migrating {agent_name} failed with: {e}")

    if candidates == 0:
        print(f"No migration candidates found ({len(agent_folders)} agent folders total)")
    else:
        print(f"Migrated {count}/{candidates} migration targets ({len(agent_folders)} agent folders total)")
