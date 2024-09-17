from memgpt.schemas.agent import CreateAgent
from memgpt.server.stateless_server import create_agent, get_db, init_db

## NOTE: probably makes sense to start with just fixing MS, and leaving agent_store for later (too many changes)
# so basically, dont touch the agent for now

# def test_persistence_manager():
#
#    session = next(get_db())
#
#    conn = SQLLiteStorageConnector(table_type=TableType.ARCHIVAL_MEMORY, config=MemGPTConfig(), user_id="test_user")


def test_create_agent():
    session = next(get_db())

    # initialize database
    init_db(session)

    # create an agent
    agent = create_agent(session, CreateAgent(name="test_agent"))
    print(agent)
