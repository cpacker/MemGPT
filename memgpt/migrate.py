import configparser
import datetime
import os
import pickle
import glob
import sys
import traceback
import uuid
import json
import shutil

import typer
from tqdm import tqdm
import questionary

from llama_index import (
    StorageContext,
    load_index_from_storage,
)

from memgpt.agent import Agent
from memgpt.data_types import AgentState, User, Passage, Source
from memgpt.metadata import MetadataStore
from memgpt.utils import MEMGPT_DIR, version_less_than, OpenAIBackcompatUnpickler, annotate_message_json_list_with_tool_calls
from memgpt.config import MemGPTConfig
from memgpt.cli.cli_config import configure
from memgpt.agent_store.storage import StorageConnector, TableType

# This is the version where the breaking change was made
VERSION_CUTOFF = "0.2.12"

# Migration backup dir (where we'll dump old agents that we successfully migrated)
MIGRATION_BACKUP_FOLDER = "migration_backups"


def wipe_config_and_reconfigure(run_configure=True):
    """Wipe (backup) the config file, and launch `memgpt configure`"""

    if not os.path.exists(os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER)):
        os.makedirs(os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER))
        os.makedirs(os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER, "agents"))

    # Get the current timestamp in a readable format (e.g., YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the new backup directory name with the timestamp
    backup_filename = os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER, f"config_backup_{timestamp}")
    existing_filename = os.path.join(MEMGPT_DIR, "config")

    # Check if the existing file exists before moving
    if os.path.exists(existing_filename):
        # shutil should work cross-platform
        shutil.move(existing_filename, backup_filename)
        typer.secho(f"Deleted config file ({existing_filename}) and saved as backup ({backup_filename})", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Couldn't find an existing config file to delete", fg=typer.colors.RED)

    if run_configure:
        # Either run configure
        configure()
    else:
        # Or create a new config with defaults
        MemGPTConfig.load()


def config_is_compatible(allow_empty=False, echo=False) -> bool:
    """Check if the config is OK to use with 0.2.12, or if it needs to be deleted"""
    # NOTE: don't use built-in load(), since that will apply defaults
    # memgpt_config = MemGPTConfig.load()
    memgpt_config_file = os.path.join(MEMGPT_DIR, "config")
    if not os.path.exists(memgpt_config_file):
        return True if allow_empty else False
    parser = configparser.ConfigParser()
    parser.read(memgpt_config_file)

    if "version" in parser and "memgpt_version" in parser["version"]:
        version = parser["version"]["memgpt_version"]
    else:
        version = None

    if version is None:
        if echo:
            typer.secho(f"Current config version is missing", fg=typer.colors.RED)
        return False
    elif version_less_than(version, VERSION_CUTOFF):
        if echo:
            typer.secho(f"Current config version ({version}) is older than migration cutoff ({VERSION_CUTOFF})", fg=typer.colors.RED)
        return False
    else:
        if echo:
            typer.secho(f"Current config version {version} is compatible!", fg=typer.colors.GREEN)
        return True


def agent_is_migrateable(agent_name: str) -> bool:
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


def migrate_source(source_name: str):
    """
    Migrate an old source folder (`~/.memgpt/sources/{source_name}`).
    """

    # 1. Load the VectorIndex from ~/.memgpt/sources/{source_name}/index
    # TODO
    source_path = os.path.join(MEMGPT_DIR, "archival", source_name, "nodes.pkl")
    assert os.path.exists(source_path), f"Source {source_name} does not exist at {source_path}"

    # load state from old checkpoint file
    from memgpt.cli.cli_load import load_index

    # 2. Create a new AgentState using the agent config + agent internal state
    config = MemGPTConfig.load()

    # gets default user
    ms = MetadataStore(config)
    user_id = uuid.UUID(config.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        raise ValueError(
            f"Failed to load user {str(user_id)} from database. Please make sure to migrate your config before migrating agents."
        )

    # insert source into metadata store
    source = Source(user_id=user.id, name=source_name)
    ms.create_source(source)

    try:
        nodes = pickle.load(open(source_path, "rb"))
        passages = []
        for node in nodes:
            # print(len(node.embedding))
            # TODO: make sure embedding config matches embedding size?
            passages.append(Passage(user_id=user.id, data_source=source_name, text=node.text, embedding=node.embedding))

        assert len(passages) > 0, f"Source {source_name} has no passages"
        conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config=config, user_id=user_id)
        conn.insert_many(passages)
        print(f"Inserted {len(passages)} to {source_name}")
    except Exception as e:
        # delete from metadata store
        ms.delete_source(source.id)
        raise ValueError(f"Failed to migrate {source_name}: {str(e)}")

    # basic checks
    source = ms.get_source(user_id=user.id, source_name=source_name)
    assert source is not None, f"Failed to load source {source_name} from database after migration"


def migrate_agent(agent_name: str):
    """Migrate an old agent folder (`~/.memgpt/agents/{agent_name}`)

    Steps:
    1. Load the agent state JSON from the old folder
    2. Create a new AgentState using the agent config + agent internal state
    3. Instantiate a new Agent by passing AgentState to Agent.__init__
       (This will automatically run into a new database)
    """

    # 1. Load the agent state JSON from the old folder
    # TODO
    agent_folder = os.path.join(MEMGPT_DIR, "agents", agent_name)
    # migration_file = os.path.join(agent_folder, MIGRATION_FILE_NAME)

    # load state from old checkpoint file
    agent_ckpt_directory = os.path.join(agent_folder, "agent_state")
    json_files = glob.glob(os.path.join(agent_ckpt_directory, "*.json"))  # This will list all .json files in the current directory.
    if not json_files:
        raise ValueError(f"Cannot load {agent_name} - no saved checkpoints found in {agent_ckpt_directory}")
        # NOTE this is a soft fail, just allow it to pass
        # return

    # Sort files based on modified timestamp, with the latest file being the first.
    state_filename = max(json_files, key=os.path.getmtime)
    state_dict = json.load(open(state_filename, "r"))

    # print(state_dict.keys())
    # print(state_dict["memory"])
    # dict_keys(['model', 'system', 'functions', 'messages', 'messages_total', 'memory'])

    # load old data from the persistence manager
    persistence_filename = os.path.basename(state_filename).replace(".json", ".persistence.pickle")
    persistence_filename = os.path.join(agent_folder, "persistence_manager", persistence_filename)
    archival_filename = os.path.join(agent_folder, "persistence_manager", "index", "nodes.pkl")
    if not os.path.exists(persistence_filename):
        raise ValueError(f"Cannot load {agent_name} - no saved persistence pickle found at {persistence_filename}")

    try:
        with open(persistence_filename, "rb") as f:
            data = pickle.load(f)
    except ModuleNotFoundError as e:
        # Patch for stripped openai package
        # ModuleNotFoundError: No module named 'openai.openai_object'
        with open(persistence_filename, "rb") as f:
            unpickler = OpenAIBackcompatUnpickler(f)
            data = unpickler.load()

        from memgpt.openai_backcompat.openai_object import OpenAIObject

        def convert_openai_objects_to_dict(obj):
            if isinstance(obj, OpenAIObject):
                # Convert to dict or handle as needed
                # print(f"detected OpenAIObject on {obj}")
                return obj.to_dict_recursive()
            elif isinstance(obj, dict):
                return {k: convert_openai_objects_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_openai_objects_to_dict(v) for v in obj]
            else:
                return obj

        data = convert_openai_objects_to_dict(data)

    # data will contain:
    # print("data.keys()", data.keys())
    # manager.all_messages = data["all_messages"]
    # manager.messages = data["messages"]
    # manager.recall_memory = data["recall_memory"]

    agent_config_filename = os.path.join(agent_folder, "config.json")
    with open(agent_config_filename, "r") as fh:
        agent_config = json.load(fh)

    # 2. Create a new AgentState using the agent config + agent internal state
    config = MemGPTConfig.load()

    # gets default user
    ms = MetadataStore(config)
    user_id = uuid.UUID(config.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        raise ValueError(
            f"Failed to load user {str(user_id)} from database. Please make sure to migrate your config before migrating agents."
        )
    #    ms.create_user(User(id=user_id))
    #    user = ms.get_user(user_id=user_id)
    #    if user is None:
    #        typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
    #        sys.exit(1)

    agent_state = AgentState(
        name=agent_config["name"],
        user_id=user.id,
        persona=agent_config["persona"],  # eg 'sam_pov'
        human=agent_config["human"],  # eg 'basic'
        preset=agent_config["preset"],  # eg 'memgpt_chat'
        state=dict(
            human=state_dict["memory"]["human"],
            persona=state_dict["memory"]["persona"],
            system=state_dict["system"],
            functions=state_dict["functions"],  # this shouldn't matter, since Agent.__init__ will re-link
            messages=annotate_message_json_list_with_tool_calls(state_dict["messages"]),
        ),
        llm_config=user.default_llm_config,
        embedding_config=user.default_embedding_config,
    )

    # 3. Instantiate a new Agent by passing AgentState to Agent.__init__
    # NOTE: the Agent.__init__ will trigger a save, which will write to the DB
    try:
        agent = Agent(
            agent_state=agent_state,
            messages_total=state_dict["messages_total"],  # TODO: do we need this?
            interface=None,
        )
    except Exception as e:
        # if "Agent with name" in str(e):
        #     print(e)
        #     return
        # elif "was specified in agent.state.functions":
        #     print(e)
        #     return
        # else:
        # raise
        raise

    # Wrap the rest in a try-except so that we can cleanup by deleting the agent if we fail
    try:
        ## 4. Insert into recall
        # TODO should this be 'messages', or 'all_messages'?
        # all_messages in recall will have fields "timestamp" and "message"
        full_message_history_buffer = annotate_message_json_list_with_tool_calls([d["message"] for d in data["all_messages"]])
        # We want to keep the timestamp
        for i in range(len(data["all_messages"])):
            data["all_messages"][i]["message"] = full_message_history_buffer[i]
        messages_to_insert = [agent.persistence_manager.json_to_message(msg) for msg in data["all_messages"]]
        agent.persistence_manager.recall_memory.insert_many(messages_to_insert)
        # print("Finished migrating recall memory")

        # TODO should we also assign data["messages"] to RecallMemory.messages?

        # 5. Insert into archival
        if os.path.exists(archival_filename):
            nodes = pickle.load(open(archival_filename, "rb"))
            passages = []
            for node in nodes:
                # print(len(node.embedding))
                # TODO: make sure embeding size matches embedding config?
                passages.append(Passage(user_id=user.id, agent_id=agent_state.id, text=node.text, embedding=node.embedding))
            if len(passages) > 0:
                agent.persistence_manager.archival_memory.storage.insert_many(passages)
                print(f"Inserted {len(passages)} passages into archival memory")

        else:
            print("No archival memory found at", archival_filename)

    except:
        ms.delete_agent(agent_state.id)
        raise

    try:
        new_agent_folder = os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER, "agents", agent_name)
        shutil.move(agent_folder, new_agent_folder)
    except Exception as e:
        print(f"Failed to move agent folder from {agent_folder} to {new_agent_folder}")
        raise


# def migrate_all_agents(stop_on_fail=True):
def migrate_all_agents(stop_on_fail: bool = False) -> dict:
    """Scan over all agent folders in MEMGPT_DIR and migrate each agent."""

    if not os.path.exists(os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER)):
        os.makedirs(os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER))
        os.makedirs(os.path.join(MEMGPT_DIR, MIGRATION_BACKUP_FOLDER, "agents"))

    if not config_is_compatible(echo=True):
        typer.secho(f"Your current config file is incompatible with MemGPT versions >= {VERSION_CUTOFF}", fg=typer.colors.RED)
        if questionary.confirm(
            "To migrate old MemGPT agents, you must delete your config file and run `memgpt configure`. Would you like to proceed?"
        ).ask():
            try:
                wipe_config_and_reconfigure()
            except Exception as e:
                typer.secho(f"Fresh config generation failed - error:\n{e}", fg=typer.colors.RED)
                raise
        else:
            typer.secho("Migration cancelled (to migrate old agents, run `memgpt migrate`)", fg=typer.colors.RED)
            raise KeyboardInterrupt()

    agents_dir = os.path.join(MEMGPT_DIR, "agents")

    # Ensure the directory exists
    if not os.path.exists(agents_dir):
        raise ValueError(f"Directory {agents_dir} does not exist.")

    # Get a list of all folders in agents_dir
    agent_folders = [f for f in os.listdir(agents_dir) if os.path.isdir(os.path.join(agents_dir, f))]

    # Iterate over each folder with a tqdm progress bar
    count = 0
    failures = []
    candidates = []
    try:
        for agent_name in tqdm(agent_folders, desc="Migrating agents"):
            # Assuming migrate_agent is a function that takes the agent name and performs migration
            try:
                if agent_is_migrateable(agent_name=agent_name):
                    candidates.append(agent_name)
                    migrate_agent(agent_name)
                    count += 1
                else:
                    continue
            except Exception as e:
                failures.append({"name": agent_name, "reason": str(e)})
                # typer.secho(f"Migrating {agent_name} failed with: {str(e)}", fg=typer.colors.RED)
                traceback.print_exc()
                if stop_on_fail:
                    raise
    except KeyboardInterrupt:
        typer.secho(f"User cancelled operation", fg=typer.colors.RED)

    if len(candidates) == 0:
        typer.secho(f"No migration candidates found ({len(agent_folders)} agent folders total)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Inspected {len(agent_folders)} agent folders")
        if len(failures) > 0:
            typer.secho(f"Failed migrations:", fg=typer.colors.RED)
            for fail in failures:
                typer.secho(f"{fail['name']}: {fail['reason']}", fg=typer.colors.RED)
            typer.secho(f"❌ {len(failures)}/{len(candidates)} migration targets failed (see reasons above)", fg=typer.colors.RED)
        if count > 0:
            typer.secho(f"✅ {count}/{len(candidates)} agents were successfully migrated to the new database format", fg=typer.colors.GREEN)

    return {
        "agent_folders": len(agent_folders),
        "migration_candidates": len(candidates),
        "successful_migrations": count,
        "failed_migrations": len(failures),
    }


def migrate_all_sources(stop_on_fail: bool = False) -> dict:
    """Scan over all agent folders in MEMGPT_DIR and migrate each agent."""

    sources_dir = os.path.join(MEMGPT_DIR, "archival")

    # Ensure the directory exists
    if not os.path.exists(sources_dir):
        raise ValueError(f"Directory {sources_dir} does not exist.")

    # Get a list of all folders in agents_dir
    source_folders = [f for f in os.listdir(sources_dir) if os.path.isdir(os.path.join(sources_dir, f))]

    # Iterate over each folder with a tqdm progress bar
    count = 0
    failures = []
    candidates = []
    try:
        for source_name in tqdm(source_folders, desc="Migrating data sources"):
            # Assuming migrate_agent is a function that takes the agent name and performs migration
            try:
                candidates.append(source_name)
                migrate_source(source_name)
                count += 1
            except Exception as e:
                failures.append({"name": source_name, "reason": str(e)})
                traceback.print_exc()
                if stop_on_fail:
                    raise
                # typer.secho(f"Migrating {agent_name} failed with: {str(e)}", fg=typer.colors.RED)
    except KeyboardInterrupt:
        typer.secho(f"User cancelled operation", fg=typer.colors.RED)

    if len(candidates) == 0:
        typer.secho(f"No migration candidates found ({len(source_folders)} source folders total)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Inspected {len(source_folders)} source folders")
        if len(failures) > 0:
            typer.secho(f"Failed migrations:", fg=typer.colors.RED)
            for fail in failures:
                typer.secho(f"{fail['name']}: {fail['reason']}", fg=typer.colors.RED)
            typer.secho(f"❌ {len(failures)}/{len(candidates)} migration targets failed (see reasons above)", fg=typer.colors.RED)
        if count > 0:
            typer.secho(f"✅ {count}/{len(candidates)} sources were successfully migrated to the new database format", fg=typer.colors.GREEN)

    return {
        "source_folders": len(source_folders),
        "migration_candidates": len(candidates),
        "successful_migrations": count,
        "failed_migrations": len(failures),
    }
