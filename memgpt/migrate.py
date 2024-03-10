import configparser
from datetime import datetime
import os
import pickle
import glob
import sys
import traceback
import uuid
import json
import shutil
from typing import Optional, List
import pytz

import typer
from tqdm import tqdm
import questionary


from memgpt.agent import Agent, save_agent
from memgpt.data_types import AgentState, User, Passage, Source, Message
from memgpt.metadata import MetadataStore
from memgpt.utils import (
    MEMGPT_DIR,
    version_less_than,
    OpenAIBackcompatUnpickler,
    annotate_message_json_list_with_tool_calls,
    parse_formatted_time,
)
from memgpt.config import MemGPTConfig
from memgpt.cli.cli_config import configure
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.persistence_manager import PersistenceManager, LocalStateManager

# This is the version where the breaking change was made
VERSION_CUTOFF = "0.2.12"

# Migration backup dir (where we'll dump old agents that we successfully migrated)
MIGRATION_BACKUP_FOLDER = "migration_backups"


def wipe_config_and_reconfigure(data_dir: str = MEMGPT_DIR, run_configure=True, create_config=True):
    """Wipe (backup) the config file, and launch `memgpt configure`"""

    if not os.path.exists(os.path.join(data_dir, MIGRATION_BACKUP_FOLDER)):
        os.makedirs(os.path.join(data_dir, MIGRATION_BACKUP_FOLDER))
        os.makedirs(os.path.join(data_dir, MIGRATION_BACKUP_FOLDER, "agents"))

    # Get the current timestamp in a readable format (e.g., YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the new backup directory name with the timestamp
    backup_filename = os.path.join(data_dir, MIGRATION_BACKUP_FOLDER, f"config_backup_{timestamp}")
    existing_filename = os.path.join(data_dir, "config")

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
    elif create_config:
        # Or create a new config with defaults
        MemGPTConfig.load()


def config_is_compatible(data_dir: str = MEMGPT_DIR, allow_empty=False, echo=False) -> bool:
    """Check if the config is OK to use with 0.2.12, or if it needs to be deleted"""
    # NOTE: don't use built-in load(), since that will apply defaults
    # memgpt_config = MemGPTConfig.load()
    memgpt_config_file = os.path.join(data_dir, "config")
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


def agent_is_migrateable(agent_name: str, data_dir: str = MEMGPT_DIR) -> bool:
    """Determine whether or not the agent folder is a migration target"""
    agent_folder = os.path.join(data_dir, "agents", agent_name)

    if not os.path.exists(agent_folder):
        raise ValueError(f"Folder {agent_folder} does not exist")

    agent_config_file = os.path.join(agent_folder, "config.json")
    if not os.path.exists(agent_config_file):
        raise ValueError(f"Agent folder {agent_folder} does not have a config file")

    try:
        with open(agent_config_file, "r", encoding="utf-8") as fh:
            agent_config = json.load(fh)
    except Exception as e:
        raise ValueError(f"Failed to load agent config file ({agent_config_file}), error = {e}")

    if not hasattr(agent_config, "memgpt_version") or version_less_than(agent_config.memgpt_version, VERSION_CUTOFF):
        return True
    else:
        return False


def migrate_source(source_name: str, data_dir: str = MEMGPT_DIR, ms: Optional[MetadataStore] = None):
    """
    Migrate an old source folder (`~/.memgpt/sources/{source_name}`).
    """

    # 1. Load the VectorIndex from ~/.memgpt/sources/{source_name}/index
    # TODO
    source_path = os.path.join(data_dir, "archival", source_name, "nodes.pkl")
    assert os.path.exists(source_path), f"Source {source_name} does not exist at {source_path}"

    # load state from old checkpoint file
    from memgpt.cli.cli_load import load_index

    # 2. Create a new AgentState using the agent config + agent internal state
    config = MemGPTConfig.load()
    if ms is None:
        ms = MetadataStore(config)

    # gets default user
    user_id = uuid.UUID(config.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        ms.create_user(User(id=user_id))
        user = ms.get_user(user_id=user_id)
        if user is None:
            typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
            sys.exit(1)
        # raise ValueError(
        # f"Failed to load user {str(user_id)} from database. Please make sure to migrate your config before migrating agents."
        # )

    # insert source into metadata store
    source = Source(user_id=user.id, name=source_name)
    ms.create_source(source)

    try:
        try:
            nodes = pickle.load(open(source_path, "rb"))
        except ModuleNotFoundError as e:
            if "No module named 'llama_index.schema'" in str(e):
                # cannot load source at all, so throw error
                raise ValueError(
                    "Failed to load archival memory due thanks to llama_index's breaking changes. Please downgrade to MemGPT version 0.3.3 or earlier to migrate this agent."
                )
            else:
                raise e

        passages = []
        for node in nodes:
            # print(len(node.embedding))
            # TODO: make sure embedding config matches embedding size?
            if len(node.embedding) != config.default_embedding_config.embedding_dim:
                raise ValueError(
                    f"Cannot migrate source {source_name} due to incompatible embedding dimentions. Please re-load this source with `memgpt load`."
                )
            passages.append(
                Passage(
                    user_id=user.id,
                    data_source=source_name,
                    text=node.text,
                    embedding=node.embedding,
                    embedding_dim=config.default_embedding_config.embedding_dim,
                    embedding_model=config.default_embedding_config.embedding_model,
                )
            )

        assert len(passages) > 0, f"Source {source_name} has no passages"
        conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config=config, user_id=user_id)
        conn.insert_many(passages)
        # print(f"Inserted {len(passages)} to {source_name}")
    except Exception as e:
        # delete from metadata store
        ms.delete_source(source.id)
        raise ValueError(f"Failed to migrate {source_name}: {str(e)}")

    # basic checks
    source = ms.get_source(user_id=user.id, source_name=source_name)
    assert source is not None, f"Failed to load source {source_name} from database after migration"


def migrate_agent(agent_name: str, data_dir: str = MEMGPT_DIR, ms: Optional[MetadataStore] = None) -> List[str]:
    """Migrate an old agent folder (`~/.memgpt/agents/{agent_name}`)

    Steps:
    1. Load the agent state JSON from the old folder
    2. Create a new AgentState using the agent config + agent internal state
    3. Instantiate a new Agent by passing AgentState to Agent.__init__
       (This will automatically run into a new database)

    If success, returns empty list
    If warning, returns a list of strings (warning message)
    If error, raises an Exception
    """
    warnings = []

    # 1. Load the agent state JSON from the old folder
    # TODO
    agent_folder = os.path.join(data_dir, "agents", agent_name)
    # migration_file = os.path.join(agent_folder, MIGRATION_FILE_NAME)

    # load state from old checkpoint file
    agent_ckpt_directory = os.path.join(agent_folder, "agent_state")
    json_files = glob.glob(os.path.join(agent_ckpt_directory, "*.json"))  # This will list all .json files in the current directory.
    if not json_files:
        raise ValueError(f"Cannot load {agent_name} - no saved checkpoints found in {agent_ckpt_directory}")
        # NOTE this is a soft fail, just allow it to pass
        # return
        # return [f"Cannot load {agent_name} - no saved checkpoints found in {agent_ckpt_directory}"]

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
        # return [f"Cannot load {agent_name} - no saved persistence pickle found at {persistence_filename}"]

    try:
        with open(persistence_filename, "rb") as f:
            data = pickle.load(f)
    except ModuleNotFoundError:
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
    with open(agent_config_filename, "r", encoding="utf-8") as fh:
        agent_config = json.load(fh)

    # 2. Create a new AgentState using the agent config + agent internal state
    config = MemGPTConfig.load()
    if ms is None:
        ms = MetadataStore(config)

    # gets default user
    user_id = uuid.UUID(config.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        ms.create_user(User(id=user_id))
        user = ms.get_user(user_id=user_id)
        if user is None:
            typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
            sys.exit(1)
        # raise ValueError(
        #     f"Failed to load user {str(user_id)} from database. Please make sure to migrate your config before migrating agents."
        # )
    #    ms.create_user(User(id=user_id))
    #    user = ms.get_user(user_id=user_id)
    #    if user is None:
    #        typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
    #        sys.exit(1)

    # create an agent_id ahead of time
    agent_id = uuid.uuid4()

    # create all the Messages in the database
    # message_objs = []
    # for message_dict in annotate_message_json_list_with_tool_calls(state_dict["messages"]):
    #     message_obj = Message.dict_to_message(
    #         user_id=user.id,
    #         agent_id=agent_id,
    #         openai_message_dict=message_dict,
    #         model=state_dict["model"] if "model" in state_dict else None,
    #         # allow_functions_style=False,
    #         allow_functions_style=True,
    #     )
    #     message_objs.append(message_obj)

    agent_state = AgentState(
        id=agent_id,
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
            # messages=[str(m.id) for m in message_objs],  # this is a list of uuids, not message dicts
        ),
        llm_config=config.default_llm_config,
        embedding_config=config.default_embedding_config,
    )

    persistence_manager = LocalStateManager(agent_state=agent_state)

    # First clean up the recall message history to add tool call ids
    # allow_tool_roles in case some of the old messages were actually already in tool call format (for whatever reason)
    full_message_history_buffer = annotate_message_json_list_with_tool_calls(
        [d["message"] for d in data["all_messages"]], allow_tool_roles=True
    )
    for i in range(len(data["all_messages"])):
        data["all_messages"][i]["message"] = full_message_history_buffer[i]

    # Figure out what messages in recall are in-context, and which are out-of-context
    agent_message_cache = state_dict["messages"]
    recall_message_full = data["all_messages"]

    def messages_are_equal(msg1, msg2):
        if msg1["role"] != msg2["role"]:
            return False
        if msg1["content"] != msg2["content"]:
            return False
        if "function_call" in msg1 and "function_call" in msg2 and msg1["function_call"] != msg2["function_call"]:
            return False
        if "name" in msg1 and "name" in msg2 and msg1["name"] != msg2["name"]:
            return False

        # otherwise checks pass, ~= equal
        return True

    in_context_messages = []
    out_of_context_messages = []
    assert len(agent_message_cache) <= len(recall_message_full), (len(agent_message_cache), len(recall_message_full))
    for i, d in enumerate(recall_message_full):
        # unpack into "timestamp" and "message"
        recall_message = d["message"]
        recall_timestamp = str(d["timestamp"])
        try:
            recall_datetime = parse_formatted_time(recall_timestamp.strip()).astimezone(pytz.utc)
        except ValueError:
            recall_datetime = datetime.strptime(recall_timestamp.strip(), "%Y-%m-%d %I:%M:%S %p").astimezone(pytz.utc)

        # message object
        message_obj = Message.dict_to_message(
            created_at=recall_datetime,
            user_id=user.id,
            agent_id=agent_id,
            openai_message_dict=recall_message,
            allow_functions_style=True,
        )

        # message is either in-context, or out-of-context

        if i >= (len(recall_message_full) - len(agent_message_cache)):
            # there are len(agent_message_cache) total messages on the agent
            # this will correspond to the last N messages in the recall memory (though possibly out-of-order)
            message_is_in_context = [messages_are_equal(recall_message, cache_message) for cache_message in agent_message_cache]
            # assert sum(message_is_in_context) <= 1, message_is_in_context
            # if any(message_is_in_context):
            #     in_context_messages.append(message_obj)
            # else:
            #     out_of_context_messages.append(message_obj)

            if not any(message_is_in_context):
                # typer.secho(
                #     f"Warning: didn't find late buffer recall message (i={i}/{len(recall_message_full)-1}) inside agent context\n{recall_message}",
                #     fg=typer.colors.RED,
                # )
                warnings.append(
                    f"Didn't find late buffer recall message (i={i}/{len(recall_message_full)-1}) inside agent context\n{recall_message}"
                )
                out_of_context_messages.append(message_obj)
            else:
                if sum(message_is_in_context) > 1:
                    # typer.secho(
                    #     f"Warning: found multiple occurences of recall message (i={i}/{len(recall_message_full)-1}) inside agent context\n{recall_message}",
                    #     fg=typer.colors.RED,
                    # )
                    warnings.append(
                        f"Found multiple occurences of recall message (i={i}/{len(recall_message_full)-1}) inside agent context\n{recall_message}"
                    )
                in_context_messages.append(message_obj)

        else:
            # if we're not in the final portion of the recall memory buffer, then it's 100% out-of-context
            out_of_context_messages.append(message_obj)

    assert len(in_context_messages) > 0, f"Couldn't find any in-context messages (agent_cache = {len(agent_message_cache)})"
    # assert len(in_context_messages) == len(agent_message_cache), (len(in_context_messages), len(agent_message_cache))
    if len(in_context_messages) != len(agent_message_cache):
        # typer.secho(
        #     f"Warning: uneven match of new in-context messages vs loaded cache ({len(in_context_messages)} != {len(agent_message_cache)})",
        #     fg=typer.colors.RED,
        # )
        warnings.append(
            f"Uneven match of new in-context messages vs loaded cache ({len(in_context_messages)} != {len(agent_message_cache)})"
        )
    # assert (
    # len(in_context_messages) + len(out_of_context_messages) == state_dict["messages_total"]
    # ), f"{len(in_context_messages)} + {len(out_of_context_messages)} != {state_dict['messages_total']}"

    # Now we can insert the messages into the actual recall database
    # So when we construct the agent from the state, they will be available
    persistence_manager.recall_memory.insert_many(out_of_context_messages)
    persistence_manager.recall_memory.insert_many(in_context_messages)

    # Overwrite the agent_state message object
    agent_state.state["messages"] = [str(m.id) for m in in_context_messages]  # this is a list of uuids, not message dicts

    ## 4. Insert into recall
    # TODO should this be 'messages', or 'all_messages'?
    # all_messages in recall will have fields "timestamp" and "message"
    # full_message_history_buffer = annotate_message_json_list_with_tool_calls([d["message"] for d in data["all_messages"]])
    # We want to keep the timestamp
    # for i in range(len(data["all_messages"])):
    # data["all_messages"][i]["message"] = full_message_history_buffer[i]
    # messages_to_insert = [
    #     Message.dict_to_message(
    #         user_id=user.id,
    #         agent_id=agent_id,
    #         openai_message_dict=msg,
    #         allow_functions_style=True,
    #     )
    #     # for msg in data["all_messages"]
    #     for msg in full_message_history_buffer
    # ]
    # agent.persistence_manager.recall_memory.insert_many(messages_to_insert)
    # print("Finished migrating recall memory")

    # 3. Instantiate a new Agent by passing AgentState to Agent.__init__
    # NOTE: the Agent.__init__ will trigger a save, which will write to the DB
    try:
        agent = Agent(
            agent_state=agent_state,
            # messages_total=state_dict["messages_total"],  # TODO: do we need this?
            messages_total=len(in_context_messages) + len(out_of_context_messages),
            interface=None,
        )
        save_agent(agent, ms=ms)
    except Exception:
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
        # TODO should we also assign data["messages"] to RecallMemory.messages?

        # 5. Insert into archival
        if os.path.exists(archival_filename):
            try:
                nodes = pickle.load(open(archival_filename, "rb"))
            except ModuleNotFoundError as e:
                if "No module named 'llama_index.schema'" in str(e):
                    print(
                        "Failed to load archival memory due thanks to llama_index's breaking changes. Please downgrade to MemGPT version 0.3.3 or earlier to migrate this agent."
                    )
                    nodes = []
                else:
                    raise e

            passages = []
            failed_inserts = []
            for node in nodes:
                if len(node.embedding) != config.default_embedding_config.embedding_dim:
                    # raise ValueError(f"Cannot migrate agent {agent_state.name} due to incompatible embedding dimentions.")
                    # raise ValueError(f"Cannot migrate agent {agent_state.name} due to incompatible embedding dimentions.")
                    failed_inserts.append(
                        f"Cannot migrate passage due to incompatible embedding dimentions ({len(node.embedding)} != {config.default_embedding_config.embedding_dim}) - content = '{node.text}'."
                    )
                passages.append(
                    Passage(
                        user_id=user.id,
                        agent_id=agent_state.id,
                        text=node.text,
                        embedding=node.embedding,
                        embedding_dim=agent_state.embedding_config.embedding_dim,
                        embedding_model=agent_state.embedding_config.embedding_model,
                    )
                )
            if len(passages) > 0:
                agent.persistence_manager.archival_memory.storage.insert_many(passages)
                # print(f"Inserted {len(passages)} passages into archival memory")

            if len(failed_inserts) > 0:
                warnings.append(
                    f"Failed to transfer {len(failed_inserts)}/{len(nodes)} passages from old archival memory: " + ", ".join(failed_inserts)
                )

        else:
            warnings.append("No archival memory found at", archival_filename)

    except:
        ms.delete_agent(agent_state.id)
        raise

    try:
        new_agent_folder = os.path.join(data_dir, MIGRATION_BACKUP_FOLDER, "agents", agent_name)
        shutil.move(agent_folder, new_agent_folder)
    except Exception:
        print(f"Failed to move agent folder from {agent_folder} to {new_agent_folder}")
        raise

    return warnings


# def migrate_all_agents(stop_on_fail=True):
def migrate_all_agents(data_dir: str = MEMGPT_DIR, stop_on_fail: bool = False, debug: bool = False) -> dict:
    """Scan over all agent folders in data_dir and migrate each agent."""

    if not os.path.exists(os.path.join(data_dir, MIGRATION_BACKUP_FOLDER)):
        os.makedirs(os.path.join(data_dir, MIGRATION_BACKUP_FOLDER))
        os.makedirs(os.path.join(data_dir, MIGRATION_BACKUP_FOLDER, "agents"))

    if not config_is_compatible(data_dir, echo=True):
        typer.secho(f"Your current config file is incompatible with MemGPT versions >= {VERSION_CUTOFF}", fg=typer.colors.RED)
        if questionary.confirm(
            "To migrate old MemGPT agents, you must delete your config file and run `memgpt configure`. Would you like to proceed?"
        ).ask():
            try:
                wipe_config_and_reconfigure(data_dir)
            except Exception as e:
                typer.secho(f"Fresh config generation failed - error:\n{e}", fg=typer.colors.RED)
                raise
        else:
            typer.secho("Migration cancelled (to migrate old agents, run `memgpt migrate`)", fg=typer.colors.RED)
            raise KeyboardInterrupt()

    agents_dir = os.path.join(data_dir, "agents")

    # Ensure the directory exists
    if not os.path.exists(agents_dir):
        raise ValueError(f"Directory {agents_dir} does not exist.")

    # Get a list of all folders in agents_dir
    agent_folders = [f for f in os.listdir(agents_dir) if os.path.isdir(os.path.join(agents_dir, f))]

    # Iterate over each folder with a tqdm progress bar
    count = 0
    successes = []  # agents that migrated w/o warnings
    warnings = []  # agents that migrated but had warnings
    failures = []  # agents that failed to migrate (fatal error)
    candidates = []
    config = MemGPTConfig.load()
    print(config)
    ms = MetadataStore(config)
    try:
        for agent_name in tqdm(agent_folders, desc="Migrating agents"):
            # Assuming migrate_agent is a function that takes the agent name and performs migration
            try:
                if agent_is_migrateable(agent_name=agent_name, data_dir=data_dir):
                    candidates.append(agent_name)
                    migration_warnings = migrate_agent(agent_name, data_dir=data_dir, ms=ms)
                    if len(migration_warnings) == 0:
                        successes.append(agent_name)
                    else:
                        warnings.append((agent_name, migration_warnings))
                    count += 1
                else:
                    continue
            except Exception as e:
                failures.append({"name": agent_name, "reason": str(e)})
                # typer.secho(f"Migrating {agent_name} failed with: {str(e)}", fg=typer.colors.RED)
                if debug:
                    traceback.print_exc()
                if stop_on_fail:
                    raise
    except KeyboardInterrupt:
        typer.secho(f"User cancelled operation", fg=typer.colors.RED)

    if len(candidates) == 0:
        typer.secho(f"No migration candidates found ({len(agent_folders)} agent folders total)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Inspected {len(agent_folders)} agent folders for migration")

        if len(warnings) > 0:
            typer.secho(f"Migration warnings:", fg=typer.colors.BRIGHT_YELLOW)
            for warn in warnings:
                typer.secho(f"{warn[0]}: {warn[1]}", fg=typer.colors.BRIGHT_YELLOW)

        if len(failures) > 0:
            typer.secho(f"Failed migrations:", fg=typer.colors.RED)
            for fail in failures:
                typer.secho(f"{fail['name']}: {fail['reason']}", fg=typer.colors.RED)

        if len(failures) > 0:
            typer.secho(
                f"🔴 {len(failures)}/{len(candidates)} agents failed to migrate (see reasons above)",
                fg=typer.colors.RED,
            )
            typer.secho(f"{[d['name'] for d in failures]}", fg=typer.colors.RED)

        if len(warnings) > 0:
            typer.secho(
                f"🟠 {len(warnings)}/{len(candidates)} agents successfully migrated with warnings (see reasons above)",
                fg=typer.colors.BRIGHT_YELLOW,
            )
            typer.secho(f"{[t[0] for t in warnings]}", fg=typer.colors.BRIGHT_YELLOW)

        if len(successes) > 0:
            typer.secho(
                f"🟢 {len(successes)}/{len(candidates)} agents successfully migrated with no warnings",
                fg=typer.colors.GREEN,
            )
            typer.secho(f"{successes}", fg=typer.colors.GREEN)

    del ms
    return {
        "agent_folders": len(agent_folders),
        "migration_candidates": candidates,
        "successful_migrations": len(successes) + len(warnings),
        "failed_migrations": failures,
        "user_id": uuid.UUID(MemGPTConfig.load().anon_clientid),
    }


def migrate_all_sources(data_dir: str = MEMGPT_DIR, stop_on_fail: bool = False, debug: bool = False) -> dict:
    """Scan over all agent folders in data_dir and migrate each agent."""

    sources_dir = os.path.join(data_dir, "archival")

    # Ensure the directory exists
    if not os.path.exists(sources_dir):
        raise ValueError(f"Directory {sources_dir} does not exist.")

    # Get a list of all folders in agents_dir
    source_folders = [f for f in os.listdir(sources_dir) if os.path.isdir(os.path.join(sources_dir, f))]

    # Iterate over each folder with a tqdm progress bar
    count = 0
    failures = []
    candidates = []
    config = MemGPTConfig.load()
    ms = MetadataStore(config)
    try:
        for source_name in tqdm(source_folders, desc="Migrating data sources"):
            # Assuming migrate_agent is a function that takes the agent name and performs migration
            try:
                candidates.append(source_name)
                migrate_source(source_name, data_dir, ms=ms)
                count += 1
            except Exception as e:
                failures.append({"name": source_name, "reason": str(e)})
                if debug:
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
            typer.secho(
                f"✅ {count}/{len(candidates)} sources were successfully migrated to the new database format", fg=typer.colors.GREEN
            )

    del ms
    return {
        "source_folders": len(source_folders),
        "migration_candidates": candidates,
        "successful_migrations": count,
        "failed_migrations": failures,
        "user_id": uuid.UUID(MemGPTConfig.load().anon_clientid),
    }
