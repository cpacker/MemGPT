from typing import Optional
import uuid
import requests
from requests import HTTPError

from memgpt.server.rest_api.admin.users import (
    CreateUserResponse,
    CreateAPIKeyResponse,
    GetAPIKeysResponse,
    DeleteAPIKeyResponse,
    DeleteUserResponse,
    GetAllUsersResponse,
)


class Admin:
    """
    Admin client allows admin-level operations on the MemGPT server.
    - Creating users
    - Generating user keys
    """

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self.headers = {"accept": "application/json", "content-type": "application/json", "authorization": f"Bearer {token}"}

    def get_users(self):
        response = requests.get(f"{self.base_url}/admin/users", headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return GetAllUsersResponse(**response.json())

    def create_key(self, user_id: uuid.UUID, key_name: str):
        payload = {"user_id": str(user_id), "key_name": key_name}
        response = requests.post(f"{self.base_url}/admin/users/keys", headers=self.headers, json=payload)
        print(response.json())
        if response.status_code != 200:
            raise HTTPError(response.json())
        return CreateAPIKeyResponse(**response.json())

    def get_keys(self, user_id: uuid.UUID):
        params = {"user_id": str(user_id)}
        response = requests.get(f"{self.base_url}/admin/users/keys", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        print(response.text, response.status_code)
        return GetAPIKeysResponse(**response.json())

    def delete_key(self, api_key: str):
        params = {"api_key": api_key}
        response = requests.delete(f"{self.base_url}/admin/users/keys", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return DeleteAPIKeyResponse(**response.json())

    def create_user(self, user_id: Optional[uuid.UUID] = None):
        payload = {"user_id": str(user_id) if user_id else None}
        response = requests.post(f"{self.base_url}/admin/users", headers=self.headers, json=payload)
        if response.status_code != 200:
            raise HTTPError(response.json())
        response_json = response.json()
        print(response_json)
        return CreateUserResponse(**response_json)

    def delete_user(self, user_id: uuid.UUID):
        params = {"user_id": str(user_id)}
        response = requests.delete(f"{self.base_url}/admin/users", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return DeleteUserResponse(**response.json())

    def _reset_server(self):
        # DANGER: this will delete all users and keys
        # clear all state associated with users
        # TODO: clear out all agents, presets, etc.
        users = self.get_users().user_list
        for user in users:
            keys = self.get_keys(user["user_id"]).api_key_list
            for key in keys:
                self.delete_key(key)
            self.delete_user(user["user_id"])
