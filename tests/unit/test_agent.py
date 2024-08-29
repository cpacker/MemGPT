from pytest import mark as m
from memgpt.schemas.memory import ChatMemory
from tests.mock_factory.models import MockAgentFactory

@m.describe("When performing basic interactions with Agents")
class TestUnitAgent:
    @m.skip()
    @m.context("and interacting with agents via Client")
    @m.it("should create an agent")
    async def test_create_agent(self, client):
        agent_name = "TestAgent"
        agent = await client.create_agent(name=agent_name)
        assert agent.name == agent_name

    @m.skip()
    @m.it("should update an agent")
    async def test_update_agent(self, client):
        agent_name = "TestAgent"
        agent = await client.create_agent(name=agent_name)

        new_system_prompt = "New system prompt"
        updated_agent = await client.update_agent(agent_id=agent.id, system=new_system_prompt)
        assert updated_agent.system == new_system_prompt

        new_memory = ChatMemory(human="New human memory", persona="New persona memory")
        updated_agent = await client.update_agent(agent_id=agent.id, memory=new_memory)
        assert updated_agent.memory.human == "New human memory"
        assert updated_agent.memory.persona == "New persona memory"

        new_message_ids = ["msg1", "msg2"]
        updated_agent = await client.update_agent(agent_id=agent.id, message_ids=new_message_ids)
        assert updated_agent.message_ids == new_message_ids

        new_tools = ["tool1", "tool2"]
        updated_agent = await client.update_agent(agent_id=agent.id, tools=new_tools)
        assert updated_agent.tools == new_tools

        new_name = "UpdatedAgentName"
        updated_agent = await client.rename_agent(agent_id=agent.id, new_name=new_name)
        assert updated_agent.name == new_name

        new_description = "Updated description"
        updated_agent = await client.update_agent(agent_id=agent.id, description=new_description)
        assert updated_agent.description == new_description


    @m.it("should delete an agent")
    async def test_delete_agent(self, client):
        agent_name = "TestAgent"
        agent = await client.create_agent(name=agent_name)
        await client.delete_agent(agent_id=agent.id)

        assert not await client.agent_exists(agent_id=agent.id)


    @m.it("should list agents")
    async def test_list_agents(self, client):
        agent_name = "TestAgent"
        agent = await client.create_agent(name=agent_name)
        agents = await client.list_agents()
        assert any(a["id"] == agent.id for a in agents)

    @m.skip()
    @m.context("and a message is sent to the agent")
    @m.it("should get the message and respond")
    async def test_send_message(self, client):
        agent_name = "TestAgentSendMessage"
        agent = await client.create_agent(name=agent_name)

        response = await client.send_message(agent_id=agent.id, message="Hello", role="user")

        assert "messages" in response
        assert "usage" in response

        assert len(response["messages"]) > 1