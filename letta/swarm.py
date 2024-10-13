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
        # metadata
        description: Optional[str] = None,
    ) -> AgentState:

        # todo: process tools for agent handoff

        agent = AgentState(
            name=name,
            agent_type=agent_type,
            embedding_config=embedding_config,
            llm_config=llm_config,
            system=system,
            tools=tools,
            description=description,
        )
        self.agents.append(agent)

        return agent

    def run(self, agent_name: str, message: str):

        history = []
        while True:
            # send message to agent
            agent_id = self.client.get_agent_id(agent_name)

            # TODO: implement with sending multiple messages
            response = self.client.send_message(agent_id=agent_id, message=message)

            # update history
            history += response.messages
