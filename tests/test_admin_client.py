import threading
import time
import uuid

import pytest

from memgpt import Admin
from tests.test_client import _reset_config, run_server

test_base_url = "http://localhost:8283"

# admin credentials
test_server_token = "test_server_token"


@pytest.fixture(scope="session", autouse=True)
def start_uvicorn_server():
    """Starts Uvicorn server in a background thread."""

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print("Starting server...")
    time.sleep(5)
    yield


@pytest.fixture(scope="module")
def admin_client():
    # Setup: Create a user via the client before the tests

    admin = Admin(test_base_url, test_server_token)
    admin._reset_server()
    yield admin


def test_admin_client(admin_client):
    _reset_config()

    # create a user
    user_id = uuid.uuid4()
    create_user1_response = admin_client.create_user(user_id)
    assert user_id == create_user1_response.user_id, f"Expected {user_id}, got {create_user1_response.user_id}"

    # create another user
    create_user_2_response = admin_client.create_user()

    # create keys
    key1_name = "test_key1"
    key2_name = "test_key2"
    create_key1_response = admin_client.create_key(user_id, key1_name)
    create_key2_response = admin_client.create_key(create_user_2_response.user_id, key2_name)

    # list users
    users = admin_client.get_users()
    assert len(users.user_list) == 2
    print(users.user_list)
    assert user_id in [uuid.UUID(u["user_id"]) for u in users.user_list]

    # list keys
    user1_keys = admin_client.get_keys(user_id)
    assert len(user1_keys.api_key_list) == 2, f"Expected 2 keys, got {user1_keys}"
    assert create_key1_response.api_key in user1_keys.api_key_list, f"Expected {create_key1_response.api_key} in {user1_keys.api_key_list}"
    assert (
        create_user1_response.api_key in user1_keys.api_key_list
    ), f"Expected {create_user1_response.api_key} in {user1_keys.api_key_list}"

    # delete key
    delete_key1_response = admin_client.delete_key(create_key1_response.api_key)
    assert delete_key1_response.api_key_deleted == create_key1_response.api_key
    assert len(admin_client.get_keys(user_id).api_key_list) == 1
    delete_key2_response = admin_client.delete_key(create_key2_response.api_key)
    assert delete_key2_response.api_key_deleted == create_key2_response.api_key
    assert len(admin_client.get_keys(create_user_2_response.user_id).api_key_list) == 1

    # delete users
    delete_user1_response = admin_client.delete_user(user_id)
    assert delete_user1_response.user_id_deleted == user_id
    delete_user2_response = admin_client.delete_user(create_user_2_response.user_id)
    assert delete_user2_response.user_id_deleted == create_user_2_response.user_id

    # list users
    users = admin_client.get_users()
    assert len(users.user_list) == 0, f"Expected 0 users, got {users}"
