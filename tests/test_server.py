import memgpt.utils as utils

utils.DEBUG = True
from memgpt.server.server import SyncServer


def test_server():
    user_id = "NULL"
    agent_id = "agent_26"

    server = SyncServer()

    try:
        server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    except ValueError as e:
        print(e)
    except:
        raise

    try:
        server.user_message(user_id=user_id, agent_id=agent_id, message="/memory")
    except ValueError as e:
        print(e)
    except:
        raise

    try:
        print(server.run_command(user_id=user_id, agent_id=agent_id, command="/memory"))
    except ValueError as e:
        print(e)
    except:
        raise

    try:
        server.user_message(user_id=user_id, agent_id="agent no-exist", message="Hello?")
    except ValueError as e:
        print(e)
    except:
        raise


if __name__ == "__main__":
    test_server()
