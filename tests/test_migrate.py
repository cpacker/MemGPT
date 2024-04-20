import os
import shutil
import uuid

from memgpt.migrate import migrate_all_agents
from memgpt.server.server import SyncServer

from .utils import create_config, wipe_config


def test_migrate_0211():
    wipe_config()
    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
    else:
        create_config("memgpt_hosted")

    data_dir = "tests/data/memgpt-0.2.11"
    tmp_dir = f"tmp_{str(uuid.uuid4())}"
    shutil.copytree(data_dir, tmp_dir)
    print("temporary directory:", tmp_dir)
    # os.environ["MEMGPT_CONFIG_PATH"] = os.path.join(data_dir, "config")
    # print(f"MEMGPT_CONFIG_PATH={os.environ['MEMGPT_CONFIG_PATH']}")
    try:
        agent_res = migrate_all_agents(tmp_dir, debug=True)
        assert len(agent_res["failed_migrations"]) == 0, f"Failed migrations: {agent_res}"

        # NOTE: source tests had to be removed since it is no longer possible to migrate llama index vector indices
        # source_res = migrate_all_sources(tmp_dir)
        # assert len(source_res["failed_migrations"]) == 0, f"Failed migrations: {source_res}"

        # TODO: assert everything is in the DB

        server = SyncServer()
        for agent_name in agent_res["migration_candidates"]:
            if agent_name not in agent_res["failed_migrations"]:
                # assert agent data exists
                agent_state = server.ms.get_agent(agent_name=agent_name, user_id=agent_res["user_id"])
                assert agent_state is not None, f"Missing agent {agent_name}"

                # assert in context messages exist
                message_ids = server.get_in_context_message_ids(user_id=agent_res["user_id"], agent_id=agent_state.id)
                assert len(message_ids) > 0

                # assert recall memories exist
                messages = server.get_agent_messages(
                    user_id=agent_state.user_id,
                    agent_id=agent_state.id,
                    start=0,
                    count=1000,
                )
                assert len(messages) > 0

        # for source_name in source_res["migration_candidates"]:
        #    if source_name not in source_res["failed_migrations"]:
        #        # assert source data exists
        #        source = server.ms.get_source(source_name=source_name, user_id=source_res["user_id"])
        #        assert source is not None
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(tmp_dir)
