import random
import string

from locust import HttpUser, between, task

from memgpt.constants import BASE_TOOLS, DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.schemas.agent import AgentState, CreateAgent
from memgpt.schemas.memgpt_request import MemGPTRequest
from memgpt.schemas.memgpt_response import MemGPTResponse
from memgpt.schemas.memory import ChatMemory
from memgpt.schemas.message import MessageCreate, MessageRole
from memgpt.utils import get_human_text, get_persona_text


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

        # generate random name
        name = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        request = CreateAgent(
            name=f"Agent-{name}",
            tools=BASE_TOOLS,
            memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        )

        # create an agent
        with self.client.post("/api/agents", json=request.model_dump(), headers=self.client.headers, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed to create agent: {response.text}")

            response_json = response.json()
            agent_state = AgentState(**response_json)
            self.agent_id = agent_state.id
            print("Created agent", self.agent_id, agent_state.name)

    @task(1)
    def send_message(self):
        messages = [MessageCreate(role=MessageRole("user"), text="hello")]
        request = MemGPTRequest(messages=messages, stream_steps=False, stream_tokens=False, return_message_object=False)

        with self.client.post(
            f"/api/agents/{self.agent_id}/messages", json=request.model_dump(), headers=self.client.headers, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed to send message: {response.text}")

            response = MemGPTResponse(**response.json())
            print("Response", response.usage)

    # @task(1)
    # def send_message_stream(self):

    #    messages = [MessageCreate(role=MessageRole("user"), text="hello")]
    #    request = MemGPTRequest(messages=messages, stream_steps=True, stream_tokens=True, return_message_object=True)
    #    if stream_tokens or stream_steps:
    #        from memgpt.client.streaming import _sse_post

    #        request.return_message_object = False
    #        return _sse_post(f"{self.base_url}/api/agents/{agent_id}/messages", request.model_dump(), self.headers)
    #    else:
    #        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/messages", json=request.model_dump(), headers=self.headers)
    #        if response.status_code != 200:
    #            raise ValueError(f"Failed to send message: {response.text}")
    #        return MemGPTResponse(**response.json())
    #    try:
    #        response = self.memgpt_client.send_message(message="Hello, world!", agent_id=self.agent_id, role="user")
    #    except Exception as e:
    #        with self.client.get("/", catch_response=True) as response:
    #            response.failure(str(e))

    # @task(2)
    # def get_agent_state(self):
    #    try:
    #        agent_state = self.memgpt_client.get_agent(agent_id=self.agent_id)
    #    except Exception as e:
    #        with self.client.get("/", catch_response=True) as response:
    #            response.failure(str(e))

    # @task(3)
    # def get_agent_memory(self):
    #    try:
    #        memory = self.memgpt_client.get_in_context_memory(agent_id=self.agent_id)
    #    except Exception as e:
    #        with self.client.get("/", catch_response=True) as response:
    #            response.failure(str(e))


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
