import os
from memgpt.migrate import migrate_all_agents, migrate_all_sources
from memgpt import MemGPT
from memgpt.config import MemGPTConfig
from .utils import wipe_config


def test_migrate_0211():

    data_dir = "tests/data/memgpt-0.2.11"
    os.environ["MEMGPT_CONFIG_PATH"] = os.path.join(data_dir, "config")
    res = migrate_all_agents(data_dir)
    assert res["failed_migrations"] == 0, f"Failed migrations: {res}"
    res = migrate_all_sources(data_dir)
    assert res["failed_migrations"] == 0, f"Failed migrations: {res}"

    # TODO: assert everything is in the DB
