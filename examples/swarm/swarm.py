import json
from typing import List, Optional

import typer

from letta import AgentState, EmbeddingConfig, LLMConfig, create_client
from letta.schemas.agent import AgentType
from letta.schemas.memory import BasicBlockMemory, Block


class Swarm:

    def __init__(self):
        self.agents = []
        self.client = create_client()

        # shared memory block (shared section of context window accross agents)
        self.shared_memory = Block(label="human", value="")

    def create_agent(
        self,
        name: Optional[str] = None,
        # agent config
        agent_type: Optional[AgentType] = AgentType.memgpt_agent,
        # model configs
        embedding_config: EmbeddingConfig = None,
        llm_config: LLMConfig = None,
        # system
        system: Optional[str] = None,
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        # instructions
        instructions: str = "",
    ) -> AgentState:

        # todo: process tools for agent handoff
        persona_value = (
            f"You are agent with name {name}. You instructions are {instructions}"
            if len(instructions) > 0
            else f"You are agent with name {name}"
        )
        persona_block = Block(label="persona", value=persona_value)
        memory = BasicBlockMemory(blocks=[persona_block, self.shared_memory])

        agent = self.client.create_agent(
            name=name,
            agent_type=agent_type,
            embedding_config=embedding_config,
            llm_config=llm_config,
            system=system,
            tools=tools,
            include_base_tools=include_base_tools,
            memory=memory,
        )
        self.agents.append(agent)

        return agent

    def reset(self):
        # delete all agents
        for agent in self.agents:
            self.client.delete_agent(agent.id)
        for block in self.client.list_blocks():
            self.client.delete_block(block.id)

    def run(self, agent_name: str, message: str):

        history = []
        while True:
            # send message to agent
            agent_id = self.client.get_agent_id(agent_name)

            print("Messaging agent: ", agent_name)
            print("History size: ", len(history))
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
            # print(messages)

            if len(messages) < 2:
                continue

            function_call = messages[-2]
            function_return = messages[-1]
            if function_call.function_call.name == "send_message":
                # return message to use
                arg_data = json.loads(function_call.function_call.arguments)
                # print(arg_data)
                return arg_data["message"]
            else:
                # swap the agent
                return_data = json.loads(function_return.function_return)
                agent_name = return_data["message"]
                typer.secho(f"Transferring to agent: {agent_name}", fg=typer.colors.RED)
                # print("Transferring to agent", agent_name)

            print()
