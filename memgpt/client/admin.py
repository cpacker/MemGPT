from typing import Optional
import uuid
import requests

from memgpt.server.rest_api.admin.users import (
    CreateUserResponse,
    CreateAPIKeyResponse,
    GetAPIKeysResponse,
    DeleteAPIKeyResponse,
    DeleteUserResponse,
    GetAllUsersResponse,
    CreateUserRequest,
    CreateAPIKeyRequest,
    GetAPIKeysRequest,
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
        return GetAllUsersResponse.model_validate_json(response.json())

    def create_key(self, user_id: uuid.UUID, key_name: str):
        payload = {"user_id": str(user_id), "key_name": key_name}
        response = requests.post(f"{self.base_url}/admin/users/keys", headers=self.headers, json=payload)
        return CreateAPIKeyResponse.model_validate_json(response.json())

    def get_keys(self, user_id: uuid.UUID):
        response = requests.get(f"{self.base_url}/admin/users/keys/{str(user_id)}", headers=self.headers)
        return GetAPIKeysResponse.model_validate_json(response.json())

    def delete_key(self, api_key: str):
        response = requests.delete(f"{self.base_url}/admin/users/keys/{api_key}", headers=self.headers)
        return DeleteAPIKeyResponse.model_validate_json(response.json())

    def create_user(self, user_id: Optional[uuid.UUID] = None):
        payload = {"user_id": str(user_id) if user_id else None}
        response = requests.post(f"{self.base_url}/admin/users", headers=self.headers, json=payload)
        response_json = response.json()
        return CreateUserResponse.model_validate_json(response_json)

    def delete_user(self, user_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/admin/users/{str(user_id)}", headers=self.headers)
        return DeleteUserResponse.model_validate_json(response.json())
