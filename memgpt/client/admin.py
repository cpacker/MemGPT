from typing import Optional
import requests


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

    def create_user(self, user_id: Optional[str] = None):
        payload = {"user_id": str(user_id)}
        response = requests.post(f"{self.base_url}/admin/users", headers=self.headers, json=payload)
        return response.json()
