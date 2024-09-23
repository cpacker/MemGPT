# TODO: add back when messaging works

# import os
# import threading
# import time
# import uuid
#
# import pytest
# from dotenv import load_dotenv
#
# from letta import Admin, create_client
# from letta.config import LettaConfig
# from letta.credentials import LettaCredentials
# from letta.settings import settings
# from tests.utils import create_config
#
# test_agent_name = f"test_client_{str(uuid.uuid4())}"
## test_preset_name = "test_preset"
# test_agent_state = None
# client = None
#
# test_agent_state_post_message = None
# test_user_id = uuid.uuid4()
#
#
## admin credentials
# test_server_token = "test_server_token"
#
#
# def _reset_config():
#
#    # Use os.getenv with a fallback to os.environ.get
#    db_url = settings.letta_pg_uri
#
#    if os.getenv("OPENAI_API_KEY"):
#        create_config("openai")
#        credentials = LettaCredentials(
#            openai_key=os.getenv("OPENAI_API_KEY"),
#        )
#    else:  # hosted
#        create_config("letta_hosted")
#        credentials = LettaCredentials()
#
#    config = LettaConfig.load()
#
#    # set to use postgres
#    config.archival_storage_uri = db_url
#    config.recall_storage_uri = db_url
#    config.metadata_storage_uri = db_url
#    config.archival_storage_type = "postgres"
#    config.recall_storage_type = "postgres"
#    config.metadata_storage_type = "postgres"
#
#    config.save()
#    credentials.save()
#    print("_reset_config :: ", config.config_path)
#
#
# def run_server():
#
#    load_dotenv()
#
#    _reset_config()
#
#    from letta.server.rest_api.server import start_server
#
#    print("Starting server...")
#    start_server(debug=True)
#
#
## Fixture to create clients with different configurations
# @pytest.fixture(
#    params=[  # whether to use REST API server
#        {"server": True},
#        # {"server": False} # TODO: add when implemented
#    ],
#    scope="module",
# )
# def admin_client(request):
#    if request.param["server"]:
#        # get URL from enviornment
#        server_url = os.getenv("MEMGPT_SERVER_URL")
#        if server_url is None:
#            # run server in thread
#            # NOTE: must set MEMGPT_SERVER_PASS enviornment variable
#            server_url = "http://localhost:8283"
#            print("Starting server thread")
#            thread = threading.Thread(target=run_server, daemon=True)
#            thread.start()
#            time.sleep(5)
#        print("Running client tests with server:", server_url)
#        # create user via admin client
#        admin = Admin(server_url, test_server_token)
#        response = admin.create_user(test_user_id)  # Adjust as per your client's method
#
#    yield admin
#
#
# def test_concurrent_messages(admin_client):
#    # test concurrent messages
#
#    # create three
#
#    results = []
#
#    def _send_message():
#        try:
#            print("START SEND MESSAGE")
#            response = admin_client.create_user()
#            token = response.api_key
#            client = create_client(base_url=admin_client.base_url, token=token)
#            agent = client.create_agent()
#
#            print("Agent created", agent.id)
#
#            st = time.time()
#            message = "Hello, how are you?"
#            response = client.send_message(agent_id=agent.id, message=message, role="user")
#            et = time.time()
#            print(f"Message sent from {st} to {et}")
#            print(response.messages)
#            results.append((st, et))
#        except Exception as e:
#            print("ERROR", e)
#
#    threads = []
#    print("Starting threads...")
#    for i in range(5):
#        thread = threading.Thread(target=_send_message)
#        threads.append(thread)
#        thread.start()
#        print("CREATED THREAD")
#
#    print("waiting for threads to finish...")
#    for thread in threads:
#        print(thread.join())
#
#    # make sure runtime are overlapping
#    assert (results[0][0] < results[1][0] and results[0][1] > results[1][0]) or (
#        results[1][0] < results[0][0] and results[1][1] > results[0][0]
#    ), f"Threads should have overlapping runtimes {results}"
#
