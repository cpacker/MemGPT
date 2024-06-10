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
    assert len(user1_keys) == 2, f"Expected 2 keys, got {user1_keys}"
    assert create_key1_response.api_key in user1_keys, f"Expected {create_key1_response.api_key} in {user1_keys}"
    assert create_user1_response.api_key in user1_keys, f"Expected {create_user1_response.api_key} in {user1_keys}"

    # delete key
    delete_key1_response = admin_client.delete_key(create_key1_response.api_key)
    assert delete_key1_response.api_key_deleted == create_key1_response.api_key
    assert len(admin_client.get_keys(user_id)) == 1
    delete_key2_response = admin_client.delete_key(create_key2_response.api_key)
    assert delete_key2_response.api_key_deleted == create_key2_response.api_key
    assert len(admin_client.get_keys(create_user_2_response.user_id)) == 1

    # delete users
    delete_user1_response = admin_client.delete_user(user_id)
    assert delete_user1_response.user_id_deleted == user_id
    delete_user2_response = admin_client.delete_user(create_user_2_response.user_id)
    assert delete_user2_response.user_id_deleted == create_user_2_response.user_id

    # list users
    users = admin_client.get_users()
    assert len(users.user_list) == 0, f"Expected 0 users, got {users}"


def test_get_users_pagination(admin_client):
    _reset_config()

    page_size = 5
    num_users = 7
    expected_users_remainder = num_users - page_size

    # create users
    all_user_ids = []
    for i in range(num_users):

        user_id = uuid.uuid4()
        all_user_ids.append(user_id)
        key_name = "test_key" + f"{i}"

        create_user_response = admin_client.create_user(user_id)
        admin_client.create_key(create_user_response.user_id, key_name)

    # list users in page 1
    get_all_users_response1 = admin_client.get_users(limit=page_size)
    cursor1 = get_all_users_response1.cursor
    user_list1 = get_all_users_response1.user_list
    assert len(user_list1) == page_size

    # list users in page 2 using cursor
    get_all_users_response2 = admin_client.get_users(cursor1, limit=page_size)
    cursor2 = get_all_users_response2.cursor
    user_list2 = get_all_users_response2.user_list

    assert len(user_list2) == expected_users_remainder
    assert cursor1 != cursor2

    # delete users
    clean_up_users_and_keys(all_user_ids)

    # list users to check pagination with no users
    users = admin_client.get_users()
    assert len(users.user_list) == 0, f"Expected 0 users, got {users}"


def clean_up_users_and_keys(user_id_list):
    admin_client = Admin(test_base_url, test_server_token)

    # clean up all keys and users
    for user_id in user_id_list:
        keys_list = admin_client.get_keys(user_id)
        for key in keys_list:
            admin_client.delete_key(key)
        admin_client.delete_user(user_id)
