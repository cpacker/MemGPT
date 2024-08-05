from typing import List, Optional

import requests
from requests import HTTPError

from memgpt.functions.functions import parse_source_code
from memgpt.functions.schema_generator import generate_schema
from memgpt.schemas.api_key import APIKey, APIKeyCreate
from memgpt.schemas.user import User, UserCreate


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

    def get_users(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[User]:
        params = {}
        if cursor:
            params["cursor"] = str(cursor)
        if limit:
            params["limit"] = limit
        response = requests.get(f"{self.base_url}/admin/users", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return [User(**user) for user in response.json()]

    def create_key(self, user_id: str, key_name: Optional[str] = None) -> APIKey:
        request = APIKeyCreate(user_id=user_id, name=key_name)
        response = requests.post(f"{self.base_url}/admin/users/keys", headers=self.headers, json=request.model_dump())
        if response.status_code != 200:
            raise HTTPError(response.json())
        return APIKey(**response.json())

    def get_keys(self, user_id: str) -> List[APIKey]:
        params = {"user_id": str(user_id)}
        response = requests.get(f"{self.base_url}/admin/users/keys", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return [APIKey(**key) for key in response.json()]

    def delete_key(self, api_key: str) -> APIKey:
        params = {"api_key": api_key}
        response = requests.delete(f"{self.base_url}/admin/users/keys", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return APIKey(**response.json())

    def create_user(self, name: Optional[str] = None) -> User:
        request = UserCreate(name=name)
        response = requests.post(f"{self.base_url}/admin/users", headers=self.headers, json=request.model_dump())
        if response.status_code != 200:
            raise HTTPError(response.json())
        response_json = response.json()
        return User(**response_json)

    def delete_user(self, user_id: str) -> User:
        params = {"user_id": str(user_id)}
        response = requests.delete(f"{self.base_url}/admin/users", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return User(**response.json())

    def _reset_server(self):
        # DANGER: this will delete all users and keys
        # clear all state associated with users
        # TODO: clear out all agents, presets, etc.
        users = self.get_users()
        for user in users:
            keys = self.get_keys(user.id)
            for key in keys:
                self.delete_key(key.key)
            self.delete_user(user.id)

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

    def list_tools(self):
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
