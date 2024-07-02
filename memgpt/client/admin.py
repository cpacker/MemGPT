import uuid
from typing import List, Optional

import requests
from requests import HTTPError

from memgpt.functions.functions import parse_source_code
from memgpt.functions.schema_generator import generate_schema
from memgpt.server.rest_api.admin.tools import (
    CreateToolRequest,
    ListToolsResponse,
    ToolModel,
)
from memgpt.server.rest_api.admin.users import (
    CreateAPIKeyResponse,
    CreateUserResponse,
    DeleteAPIKeyResponse,
    DeleteUserResponse,
    GetAllUsersResponse,
    GetAPIKeysResponse,
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

    def get_users(self, cursor: Optional[uuid.UUID] = None, limit: Optional[int] = 50):
        payload = {"cursor": str(cursor) if cursor else None, "limit": limit}
        response = requests.get(f"{self.base_url}/admin/users", headers=self.headers, json=payload)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return GetAllUsersResponse(**response.json())

    def create_key(self, user_id: uuid.UUID, key_name: str):
        payload = {"user_id": str(user_id), "key_name": key_name}
        response = requests.post(f"{self.base_url}/admin/users/keys", headers=self.headers, json=payload)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return CreateAPIKeyResponse(**response.json())

    def get_keys(self, user_id: uuid.UUID):
        params = {"user_id": str(user_id)}
        response = requests.get(f"{self.base_url}/admin/users/keys", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return GetAPIKeysResponse(**response.json()).api_key_list

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
            keys = self.get_keys(user["user_id"])
            for key in keys:
                self.delete_key(key)
            self.delete_user(user["user_id"])

    # tools
    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        update: Optional[bool] = True,  # TODO: actually use this
        tags: Optional[List[str]] = None,
    ):
        """Create a tool
        Args:
            func (callable): The function to create a tool for.
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            update (bool, optional): Update the tool if it already exists. Defaults to True.
        Returns:
            Tool object
        """

        # TODO: check if tool already exists
        # TODO: how to load modules?
        # parse source code/schema
        source_code = parse_source_code(func)
        json_schema = generate_schema(func, name)
        source_type = "python"
        json_schema["name"]

        if "memory" in tags:
            # special modifications to memory functions
            # self.memory -> self.memory.memory, since Agent.memory.memory needs to be modified (not BaseMemory.memory)
            source_code = source_code.replace("self.memory", "self.memory.memory")

        # create data
        data = {"source_code": source_code, "source_type": source_type, "tags": tags, "json_schema": json_schema}
        CreateToolRequest(**data)  # validate

        # make REST request
        response = requests.post(f"{self.base_url}/admin/tools", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create tool: {response.text}")
        return ToolModel(**response.json())

    def list_tools(self) -> ListToolsResponse:
        response = requests.get(f"{self.base_url}/admin/tools", headers=self.headers)
        return ListToolsResponse(**response.json()).tools

    def delete_tool(self, name: str):
        response = requests.delete(f"{self.base_url}/admin/tools/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to delete tool: {response.text}")
        return response.json()

    def get_tool(self, name: str):
        response = requests.get(f"{self.base_url}/admin/tools/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return ToolModel(**response.json())
