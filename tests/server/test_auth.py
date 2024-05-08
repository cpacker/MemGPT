from httpx import AsyncClient


class TestAuth:

    def test_login(self, client: AsyncClient):
        response = client.post("/login", json={"username": "test", "password": "test"})
        assert response.status_code == 200