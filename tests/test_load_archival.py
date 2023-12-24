# import tempfile
# import asyncio
import os
import pytest
from memgpt.connectors.storage import StorageConnector, TableType

# import asyncio
from datasets import load_dataset

# import memgpt
from memgpt.cli.cli_load import load_directory, load_database, load_webpage

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

    # test: delete source
    data_source_conn.delete({"name": name})
    passages_conn.delete({"data_source": name})
    assert len(data_source_conn.get_all({"name": name})) == 0
    assert len(passages_conn.get_all({"data_source": name})) == 0
