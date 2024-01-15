import uuid

import memgpt.utils as utils

utils.DEBUG = True
from memgpt.config import MemGPTConfig
from memgpt.server.server import SyncServer
from .utils import wipe_config, wipe_memgpt_home


def test_server():
    wipe_memgpt_home()

    config = MemGPTConfig.load()
    user_id = uuid.UUID(config.anon_clientid)
    server = SyncServer()

    try:
        fake_agent_id = uuid.uuid4()
        server.user_message(user_id=user_id, agent_id=fake_agent_id, message="Hello?")
        raise Exception("user_message call should have failed")
    except (KeyError, ValueError) as e:
        # Error is expected
        print(e)
    except:
        raise

    agent_state = server.create_agent(
        user_id=user_id,
        agent_config=dict(
            preset="memgpt_chat",
            human="cs_phd",
            persona="sam_pov",
        ),
    )
    print(f"Created agent\n{agent_state}")

    try:
        server.user_message(user_id=user_id, agent_id=agent_state.id, message="/memory")
        raise Exception("user_message call should have failed")
    except ValueError as e:
        # Error is expected
        print(e)
    except:
        raise

    print(server.run_command(user_id=user_id, agent_id=agent_state.id, command="/memory"))


if __name__ == "__main__":
    test_server()
