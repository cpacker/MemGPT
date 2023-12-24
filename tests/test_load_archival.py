# import tempfile
# import asyncio
import os
import pytest
from memgpt.connectors.storage import StorageConnector, TableType

# import asyncio
from datasets import load_dataset

# import memgpt
from memgpt.cli.cli_load import load_directory, load_database, load_webpage
from memgpt.cli.cli import attach
from memgpt.constants import DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_HUMAN
from memgpt.config import AgentConfig, MemGPTConfig

# import memgpt.presets as presets
# import memgpt.personas.personas as personas
# import memgpt.humans.humans as humans
# from memgpt.persistence_manager import InMemoryStateManager, LocalStateManager

# # from memgpt.config import AgentConfig
# from memgpt.constants import MEMGPT_DIR, DEFAULT_MEMGPT_MODEL
# import memgpt.interface  # for printing to terminal


# @pytest.mark.parametrize("storage_connector", ["sqllite", "postgres"])
@pytest.mark.parametrize("metadata_storage_connector", ["sqlite"])
@pytest.mark.parametrize("passage_storage_connector", ["chroma"])
def test_load_directory(metadata_storage_connector, passage_storage_connector):

    data_source_conn = StorageConnector.get_storage_connector(storage_type=metadata_storage_connector, table_type=TableType.DATA_SOURCES)
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, storage_type=passage_storage_connector)

    # load hugging face dataset
    # dataset_name = "MemGPT/example_short_stories"
    # dataset = load_dataset(dataset_name)

    # cache_dir = os.getenv("HF_DATASETS_CACHE")
    # if cache_dir is None:
    #    # Construct the default path if the environment variable is not set.
    #    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    # print("HF Directory", cache_dir)
    name = "test_dataset"
    cache_dir = "CONTRIBUTING.md"

    # clear out data
    data_source_conn.delete_table()
    passages_conn.delete_table()
    data_source_conn = StorageConnector.get_storage_connector(storage_type=metadata_storage_connector, table_type=TableType.DATA_SOURCES)
    passages_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, storage_type=passage_storage_connector)

    # test: load directory
    load_directory(name=name, input_dir=None, input_files=[cache_dir], recursive=False)  # cache_dir,

    # test to see if contained in storage
    sources = data_source_conn.get_all({"name": name})
    assert len(sources) == 1, f"Expected 1 source, but got {len(sources)}"
    assert sources[0].name == name, f"Expected name {name}, but got {sources[0].name}"
    print("Source", sources)

    # test to see if contained in storage
    passages = passages_conn.get_all({"data_source": name})
    assert len(passages) > 0, f"Expected >0 passages, but got {len(passages)}"
    assert [p.data_source == name for p in passages]
    print("Passages", passages)

    # test: listing sources
    sources = data_source_conn.get_all()
    print("All sources", [s.name for s in sources])

    # test loading into an agent
    # create agent
    agent_config = AgentConfig(
        name="test_agent",
        persona=DEFAULT_PERSONA,
        human=DEFAULT_HUMAN,
        model=DEFAULT_MEMGPT_MODEL,
    )
    agent_config.save()
    # create storage connector
    conn = StorageConnector.get_storage_connector(
        storage_type=passage_storage_connector, table_type=TableType.ARCHIVAL_MEMORY, agent_config=agent_config
    )
    assert conn.size() == 0

    # attach data
    attach(agent=agent_config.name, data_source=name)

    # test to see if contained in storage
    assert len(passages) == conn.size()
    assert len(passages) == len(conn.get_all({"data_source": name}))

    # test: delete source
    data_source_conn.delete({"name": name})
    passages_conn.delete({"data_source": name})
    assert len(data_source_conn.get_all({"name": name})) == 0
    assert len(passages_conn.get_all({"data_source": name})) == 0
