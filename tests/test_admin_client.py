import threading
import time

import pytest

from memgpt import Admin

test_base_url = "http://localhost:8283"

# admin credentials
test_server_token = "test_server_token"


def run_server():
    from memgpt.server.rest_api.server import start_server

    print("Starting server...")
    start_server(debug=True)


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

    # create a user
    user_name = "test_user"
    user1 = admin_client.create_user(user_name)
    assert user_name == user1.name, f"Expected {user_name}, got {user1.name}"

    # create another user
    user2 = admin_client.create_user()

    # create keys
    key1_name = "test_key1"
    key2_name = "test_key2"
    api_key1 = admin_client.create_key(user1.id, key1_name)
    admin_client.create_key(user2.id, key2_name)

    # list users
    users = admin_client.get_users()
    assert len(users) == 2
    assert user1.id in [user.id for user in users]
    assert user2.id in [user.id for user in users]

    # list keys
    user1_keys = admin_client.get_keys(user1.id)
    assert len(user1_keys) == 1, f"Expected 1 keys, got {user1_keys}"
    assert api_key1.key == user1_keys[0].key

    # delete key
    deleted_key1 = admin_client.delete_key(api_key1.key)
    assert deleted_key1.key == api_key1.key
    assert len(admin_client.get_keys(user1.id)) == 0

    # delete users
    deleted_user1 = admin_client.delete_user(user1.id)
    assert deleted_user1.id == user1.id
    deleted_user2 = admin_client.delete_user(user2.id)
    assert deleted_user2.id == user2.id

    # list users
    users = admin_client.get_users()
    assert len(users) == 0, f"Expected 0 users, got {users}"


# def test_get_users_pagination(admin_client):
#
#    page_size = 5
#    num_users = 7
#    expected_users_remainder = num_users - page_size
#
#    # create users
#    all_user_ids = []
#    for i in range(num_users):
#
#        user_id = uuid.uuid4()
#        all_user_ids.append(user_id)
#        key_name = "test_key" + f"{i}"
#
#        create_user_response = admin_client.create_user(user_id)
#        admin_client.create_key(create_user_response.user_id, key_name)
#
#    # list users in page 1
#    get_all_users_response1 = admin_client.get_users(limit=page_size)
#    cursor1 = get_all_users_response1.cursor
#    user_list1 = get_all_users_response1.user_list
#    assert len(user_list1) == page_size
#
#    # list users in page 2 using cursor
#    get_all_users_response2 = admin_client.get_users(cursor1, limit=page_size)
#    cursor2 = get_all_users_response2.cursor
#    user_list2 = get_all_users_response2.user_list
#
#    assert len(user_list2) == expected_users_remainder
#    assert cursor1 != cursor2
#
#    # delete users
#    clean_up_users_and_keys(all_user_ids)
#
#    # list users to check pagination with no users
#    users = admin_client.get_users()
#    assert len(users.user_list) == 0, f"Expected 0 users, got {users}"


def clean_up_users_and_keys(user_id_list):
    admin_client = Admin(test_base_url, test_server_token)

    # clean up all keys and users
    for user_id in user_id_list:
        keys_list = admin_client.get_keys(user_id)
        for key in keys_list:
            admin_client.delete_key(key)
        admin_client.delete_user(user_id)
