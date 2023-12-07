import memgpt.utils as utils

utils.DEBUG = True
from memgpt.server.server import SyncServer


def test_server():
    user_id = "NULL"
    agent_id = "agent_26"

    server = SyncServer()

    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")

    try:
        server.user_message(user_id=user_id, agent_id=agent_id, message="/memory")
    except ValueError as e:
        print(e)
    except:
        raise

    print(server.run_command(user_id=user_id, agent_id=agent_id, command="/memory"))


if __name__ == "__main__":
    test_server()
