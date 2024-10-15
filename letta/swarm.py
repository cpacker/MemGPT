import json
from typing import List, Optional

from letta import AgentState, EmbeddingConfig, LLMConfig, create_client
from letta.schemas.agent import AgentType


class Swarm:

    def __init__(self):
        self.agents = []
        self.client = create_client()

    def create_agent(
        self,
        name: Optional[str] = None,
        # agent config
        agent_type: Optional[AgentType] = AgentType.memgpt_agent,
        # model configs
        embedding_config: EmbeddingConfig = None,
        llm_config: LLMConfig = None,
        # memory
        # memory: Memory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        # system
        system: Optional[str] = None,
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
    ) -> AgentState:

        # todo: process tools for agent handoff

        agent = self.client.create_agent(
            name=name,
            agent_type=agent_type,
            embedding_config=embedding_config,
            llm_config=llm_config,
            system=system,
            tools=tools,
            include_base_tools=include_base_tools,
        )
        self.agents.append(agent)

        return agent

    def reset(self):
        # delete all agents
        for agent in self.agents:
            self.client.delete_agent(agent.id)

    def run(self, agent_name: str, message: str):

        history = []
        while True:
            # send message to agent
            agent_id = self.client.get_agent_id(agent_name)

            print("MESSAGING AGENT", agent_name)
            print("Num messages", len(history))
            # print(self.client.get_agent(agent_id).tools)
            # TODO: implement with sending multiple messages
            if len(history) == 0:
                response = self.client.send_message(agent_id=agent_id, message=message, role="user", include_full_message=True)
            else:
                response = self.client.send_messages(agent_id=agent_id, messages=history, include_full_message=True)

            # update history
            history += response.messages

            # grab responses
            messages = []
            for message in response.messages:
                messages += message.to_letta_message()

            # get new agent (see tool call)
            print(messages)

            function_call = messages[-2]
            function_return = messages[-1]
            if function_call.function_call.name == "send_message":
                # return message to use
                arg_data = json.loads(function_call.function_call.arguments)
                print(arg_data)
                return arg_data["message"]
            else:
                # swap the agent
                return_data = json.loads(function_return.function_return)
                print(return_data)
                agent_name = return_data["message"]

            print()


def transfer_agent_b(self):
    """
    Transfer conversation to agent B.

    Returns:
        str: name of agent to transfer to
    """
    return "agentb"


def transfer_agent_a(self):
    """
    Transfer conversation to agent A.

    Returns:
        str: name of agent to transfer to
    """
    return "agenta"


swarm = Swarm()
try:
    swarm.client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))
    swarm.client.set_default_llm_config(LLMConfig.default_config(model_name="gpt-4"))
    transfer_a = swarm.client.create_tool(transfer_agent_a, terminal=True)
    transfer_b = swarm.client.create_tool(transfer_agent_b, terminal=True)
    agent_a = swarm.create_agent(name="agentb", tools=[transfer_a.name])
    agent_b = swarm.create_agent(name="agenta", tools=[transfer_b.name])

    swarm.run(agent_name="agenta", message="Transfer me to agent b by calling the transfer_agent_b tool")
except Exception as e:
    print(e)
    swarm.reset()
    raise e

swarm.reset()
