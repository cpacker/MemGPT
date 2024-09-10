import random
import string

from locust import HttpUser, between, task

from memgpt.client.client import RESTClient


class MemGPTUser(HttpUser):
    wait_time = between(1, 5)
    token = None
    agent_id = None

    def on_start(self):
        # Create a user and get the token
        self.client.headers = {"Authorization": "Bearer password"}
        user_data = {"name": f"User-{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"}
        response = self.client.post("/admin/users", json=user_data)
        response_json = response.json()
        print(response_json)
        self.user_id = response_json["id"]

        # create a token
        response = self.client.post("/admin/users/keys", json={"user_id": self.user_id})
        self.token = response.json()["key"]

        # reset to use user token as headers
        self.client.headers = {"Authorization": f"Bearer {self.token}"}

        ## Create an agent for this user
        # agent_data = {
        #    "name": f"Agent-{self.token[:8]}",
        #    "tools": BASE_TOOLS
        # }
        # response = self.client.post("/agents", json=agent_data)
        # self.agent_id = response.json()["id"]

        print("HOST", self.host)
        client = RESTClient(token=self.token, base_url=self.host)
        agent_state = client.create_agent(name=f"Agent-{self.token[:8]}")
        self.agent_id = agent_state.id
        self.memgpt_client = client
        print("Created agent", self.agent_id)

    @task(1)
    def send_message(self):
        try:
            response = self.memgpt_client.send_message(message="Hello, world!", agent_id=self.agent_id, role="user")
        except Exception as e:
            with self.client.get("/", catch_response=True) as response:
                response.failure(str(e))

    @task(2)
    def get_agent_state(self):
        try:
            agent_state = self.memgpt_client.get_agent(agent_id=self.agent_id)
        except Exception as e:
            with self.client.get("/", catch_response=True) as response:
                response.failure(str(e))

    @task(3)
    def get_agent_memory(self):
        try:
            memory = self.memgpt_client.get_in_context_memory(agent_id=self.agent_id)
        except Exception as e:
            with self.client.get("/", catch_response=True) as response:
                response.failure(str(e))


# class AdminUser(HttpUser):
#    wait_time = between(5, 10)
#    token = None
#
#    def on_start(self):
#        # Authenticate as admin
#        self.client.headers = {"Authorization": "pasword"}
#
#    @task
#    def create_user(self):
#        user_data = {
#            "name": f"User-{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
#        }
#        self.client.post("/admin/users", json=user_data)
#
#    @task
#    def get_all_users(self):
#        self.client.get("/admin/users")
#
#    @task
#    def get_all_agents(self):
#        self.client.get("/api/admin/agents")
#
