import logging
import time
from typing import Callable, Dict, Generator, List, Optional, Union

import requests

import memgpt.utils
from memgpt.config import MemGPTConfig
from memgpt.constants import BASE_TOOLS, DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.data_sources.connectors import DataConnector
from memgpt.functions.functions import parse_source_code
from memgpt.memory import get_memory_functions
from memgpt.schemas.agent import AgentState, CreateAgent, UpdateAgentState
from memgpt.schemas.block import (
    Block,
    CreateBlock,
    CreateHuman,
    CreatePersona,
    Human,
    Persona,
    UpdateBlock,
    UpdateHuman,
    UpdatePersona,
)
from memgpt.schemas.embedding_config import EmbeddingConfig

# new schemas
from memgpt.schemas.enums import JobStatus, MessageRole
from memgpt.schemas.job import Job
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.memgpt_request import MemGPTRequest
from memgpt.schemas.memgpt_response import MemGPTResponse, MemGPTStreamingResponse
from memgpt.schemas.memory import (
    ArchivalMemorySummary,
    ChatMemory,
    CreateArchivalMemory,
    Memory,
    RecallMemorySummary,
)
from memgpt.schemas.message import Message, MessageCreate, UpdateMessage
from memgpt.schemas.openai.chat_completions import ToolCall
from memgpt.schemas.passage import Passage
from memgpt.schemas.source import Source, SourceCreate, SourceUpdate
from memgpt.schemas.tool import Tool, ToolCreate, ToolUpdate
from memgpt.schemas.user import UserCreate
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.utils import get_human_text, get_persona_text


def create_client(base_url: Optional[str] = None, token: Optional[str] = None):
    if base_url is None:
        return LocalClient()
    else:
        return RESTClient(base_url, token)


class AbstractClient(object):
    def __init__(
        self,
        auto_save: bool = False,
        debug: bool = False,
    ):
        self.auto_save = auto_save
        self.debug = debug

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        raise NotImplementedError

    def create_agent(
        self,
        name: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        memory: Memory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
        description: Optional[str] = None,
    ) -> AgentState:
        raise NotImplementedError

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        message_ids: Optional[List[str]] = None,
        memory: Optional[Memory] = None,
    ):
        raise NotImplementedError

    def rename_agent(self, agent_id: str, new_name: str):
        raise NotImplementedError

    def delete_agent(self, agent_id: str):
        raise NotImplementedError

    def get_agent(self, agent_id: str) -> AgentState:
        raise NotImplementedError

    def get_agent_id(self, agent_name: str) -> AgentState:
        raise NotImplementedError

    def get_in_context_memory(self, agent_id: str) -> Memory:
        raise NotImplementedError

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        raise NotImplementedError

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        raise NotImplementedError

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        raise NotImplementedError

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        raise NotImplementedError

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        stream: Optional[bool] = False,
        include_full_message: Optional[bool] = False,
    ) -> MemGPTResponse:
        raise NotImplementedError

    def user_message(self, agent_id: str, message: str, include_full_message: Optional[bool] = False) -> MemGPTResponse:
        raise NotImplementedError

    def create_human(self, name: str, text: str) -> Human:
        raise NotImplementedError

    def create_persona(self, name: str, text: str) -> Persona:
        raise NotImplementedError

    def list_humans(self) -> List[Human]:
        raise NotImplementedError

    def list_personas(self) -> List[Persona]:
        raise NotImplementedError

    def update_human(self, human_id: str, text: str) -> Human:
        raise NotImplementedError

    def update_persona(self, persona_id: str, text: str) -> Persona:
        raise NotImplementedError

    def get_persona(self, id: str) -> Persona:
        raise NotImplementedError

    def get_human(self, id: str) -> Human:
        raise NotImplementedError

    def get_persona_id(self, name: str) -> str:
        raise NotImplementedError

    def get_human_id(self, name: str) -> str:
        raise NotImplementedError

    def delete_persona(self, id: str):
        raise NotImplementedError

    def delete_human(self, id: str):
        raise NotImplementedError

    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        update: Optional[bool] = True,
        tags: Optional[List[str]] = None,
    ) -> Tool:
        raise NotImplementedError

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        func: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
    ) -> Tool:
        raise NotImplementedError

    def list_tools(self) -> List[Tool]:
        raise NotImplementedError

    def get_tool(self, id: str) -> Tool:
        raise NotImplementedError

    def delete_tool(self, id: str):
        raise NotImplementedError

    def get_tool_id(self, name: str) -> Optional[str]:
        raise NotImplementedError

    def load_data(self, connector: DataConnector, source_name: str):
        raise NotImplementedError

    def load_file_into_source(self, filename: str, source_id: str, blocking=True) -> Job:
        raise NotImplementedError

    def create_source(self, name: str) -> Source:
        raise NotImplementedError

    def delete_source(self, source_id: str):
        raise NotImplementedError

    def get_source(self, source_id: str) -> Source:
        raise NotImplementedError

    def get_source_id(self, source_name: str) -> str:
        raise NotImplementedError

    def attach_source_to_agent(self, agent_id: str, source_id: Optional[str] = None, source_name: Optional[str] = None):
        raise NotImplementedError

    def detach_source_from_agent(self, agent_id: str, source_id: Optional[str] = None, source_name: Optional[str] = None):
        raise NotImplementedError

    def list_sources(self) -> List[Source]:
        raise NotImplementedError

    def list_attached_sources(self, agent_id: str) -> List[Source]:
        raise NotImplementedError

    def update_source(self, source_id: str, name: Optional[str] = None) -> Source:
        raise NotImplementedError

    def insert_archival_memory(self, agent_id: str, memory: str) -> List[Passage]:
        raise NotImplementedError

    def delete_archival_memory(self, agent_id: str, memory_id: str):
        raise NotImplementedError

    def get_archival_memory(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Passage]:
        raise NotImplementedError

    def get_messages(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Message]:
        raise NotImplementedError

    def list_models(self) -> List[LLMConfig]:
        raise NotImplementedError

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        raise NotImplementedError


class RESTClient(AbstractClient):
    """
    REST client for MemGPT

    Attributes:
        base_url (str): Base URL of the REST API
        headers (Dict): Headers for the REST API (includes token)
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        api_prefix: str = "v1",
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.

        Args:
            auto_save (bool): Whether to automatically save changes.
            user_id (str): The user ID.
            debug (bool): Whether to print debug information.
        """
        super().__init__(debug=debug)
        self.base_url = base_url
        self.api_prefix = api_prefix
        self.headers = {"accept": "application/json", "authorization": f"Bearer {token}"}

    def list_agents(self) -> List[AgentState]:
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents", headers=self.headers)
        return [AgentState(**agent) for agent in response.json()]

    def agent_exists(self, agent_id: str) -> bool:
        """
        Check if an agent exists

        Args:
            agent_id (str): ID of the agent
            agent_name (str): Name of the agent

        Returns:
            exists (bool): `True` if the agent exists, `False` otherwise
        """

        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}", headers=self.headers)
        if response.status_code == 404:
            # not found error
            return False
        elif response.status_code == 200:
            return True
        else:
            raise ValueError(f"Failed to check if agent exists: {response.text}")

    def create_agent(
        self,
        name: Optional[str] = None,
        # model configs
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        # memory
        memory: Memory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        # system
        system: Optional[str] = None,
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        # metadata
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
        description: Optional[str] = None,
    ) -> AgentState:
        """Create an agent

        Args:
            name (str): Name of the agent
            embedding_config (EmbeddingConfig): Embedding configuration
            llm_config (LLMConfig): LLM configuration
            memory (Memory): Memory configuration
            system (str): System configuration
            tools (List[str]): List of tools
            include_base_tools (bool): Include base tools
            metadata (Dict): Metadata
            description (str): Description

        Returns:
            agent_state (AgentState): State of the created agent
        """

        # TODO: implement this check once name lookup works
        # if name:
        #    exist_agent_id = self.get_agent_id(agent_name=name)

        #    raise ValueError(f"Agent with name {name} already exists")

        # construct list of tools
        tool_names = []
        if tools:
            tool_names += tools
        if include_base_tools:
            tool_names += BASE_TOOLS

        # add memory tools
        memory_functions = get_memory_functions(memory)
        for func_name, func in memory_functions.items():
            tool = self.create_tool(func, name=func_name, tags=["memory", "memgpt-base"], update=True)
            tool_names.append(tool.name)

        # create agent
        request = CreateAgent(
            name=name,
            description=description,
            metadata_=metadata,
            memory=memory,
            tools=tool_names,
            system=system,
            llm_config=llm_config,
            embedding_config=embedding_config,
        )

        response = requests.post(f"{self.base_url}/{self.api_prefix}/agents", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Status {response.status_code} - Failed to create agent: {response.text}")
        return AgentState(**response.json())

    def update_message(
        self,
        agent_id: str,
        message_id: str,
        role: Optional[MessageRole] = None,
        text: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        request = UpdateMessage(
            id=message_id,
            role=role,
            text=text,
            name=name,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        response = requests.patch(
            f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/messages/{message_id}", json=request.model_dump(), headers=self.headers
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to update message: {response.text}")
        return Message(**response.json())

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        message_ids: Optional[List[str]] = None,
        memory: Optional[Memory] = None,
    ):
        """
        Update an existing agent

        Args:
            agent_id (str): ID of the agent
            name (str): Name of the agent
            description (str): Description of the agent
            system (str): System configuration
            tools (List[str]): List of tools
            metadata (Dict): Metadata
            llm_config (LLMConfig): LLM configuration
            embedding_config (EmbeddingConfig): Embedding configuration
            message_ids (List[str]): List of message IDs
            memory (Memory): Memory configuration

        Returns:
            agent_state (AgentState): State of the updated agent
        """
        request = UpdateAgentState(
            id=agent_id,
            name=name,
            system=system,
            tools=tools,
            description=description,
            metadata_=metadata,
            llm_config=llm_config,
            embedding_config=embedding_config,
            message_ids=message_ids,
            memory=memory,
        )
        response = requests.patch(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update agent: {response.text}")
        return AgentState(**response.json())

    def rename_agent(self, agent_id: str, new_name: str):
        """
        Rename an agent

        Args:
            agent_id (str): ID of the agent
            new_name (str): New name for the agent

        """
        return self.update_agent(agent_id, name=new_name)

    def delete_agent(self, agent_id: str):
        """
        Delete an agent

        Args:
            agent_id (str): ID of the agent to delete
        """
        response = requests.delete(f"{self.base_url}/{self.api_prefix}/agents/{str(agent_id)}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete agent: {response.text}"

    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        """
        Get an agent's state by it's ID.

        Args:
            agent_id (str): ID of the agent

        Returns:
            agent_state (AgentState): State representation of the agent
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}", headers=self.headers)
        assert response.status_code == 200, f"Failed to get agent: {response.text}"
        return AgentState(**response.json())

    def get_agent_id(self, agent_name: str) -> AgentState:
        """
        Get the ID of an agent by name (names are unique per user)

        Args:
            agent_name (str): Name of the agent

        Returns:
            agent_id (str): ID of the agent
        """
        # TODO: implement this
        raise NotImplementedError

    # memory
    def get_in_context_memory(self, agent_id: str) -> Memory:
        """
        Get the in-contxt (i.e. core) memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            memory (Memory): In-context memory of the agent
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/memory", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get in-context memory: {response.text}")
        return Memory(**response.json())

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        """
        Update the in-context memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            memory (Memory): The updated in-context memory of the agent

        """
        memory_update_dict = {section: value}
        response = requests.patch(
            f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/memory", json=memory_update_dict, headers=self.headers
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to update in-context memory: {response.text}")
        return Memory(**response.json())

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        """
        Get a summary of the archival memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            summary (ArchivalMemorySummary): Summary of the archival memory

        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/memory/archival", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get archival memory summary: {response.text}")
        return ArchivalMemorySummary(**response.json())

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        """
        Get a summary of the recall memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            summary (RecallMemorySummary): Summary of the recall memory
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/memory/recall", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get recall memory summary: {response.text}")
        return RecallMemorySummary(**response.json())

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """
        Get in-context messages of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            messages (List[Message]): List of in-context messages
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/memory/messages", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get in-context messages: {response.text}")
        return [Message(**message) for message in response.json()]

    # agent interactions

    def user_message(self, agent_id: str, message: str, include_full_message: Optional[bool] = False) -> MemGPTResponse:
        """
        Send a message to an agent as a user

        Args:
            agent_id (str): ID of the agent
            message (str): Message to send

        Returns:
            response (MemGPTResponse): Response from the agent
        """
        return self.send_message(agent_id, message, role="user", include_full_message=include_full_message)

    def save(self):
        raise NotImplementedError

    # archival memory

    def get_archival_memory(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Passage]:
        """
        Get archival memory from an agent with pagination.

        Args:
            agent_id (str): ID of the agent
            before (str): Get memories before a certain time
            after (str): Get memories after a certain time
            limit (int): Limit number of memories

        Returns:
            passages (List[Passage]): List of passages
        """
        params = {"limit": limit}
        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{str(agent_id)}/archival", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to get archival memory: {response.text}"
        return [Passage(**passage) for passage in response.json()]

    def insert_archival_memory(self, agent_id: str, memory: str) -> List[Passage]:
        """
        Insert archival memory into an agent

        Args:
            agent_id (str): ID of the agent
            memory (str): Memory string to insert

        Returns:
            passages (List[Passage]): List of inserted passages
        """
        request = CreateArchivalMemory(text=memory)
        response = requests.post(
            f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/archival", headers=self.headers, json=request.model_dump()
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to insert archival memory: {response.text}")
        return [Passage(**passage) for passage in response.json()]

    def delete_archival_memory(self, agent_id: str, memory_id: str):
        """
        Delete archival memory from an agent

        Args:
            agent_id (str): ID of the agent
            memory_id (str): ID of the memory
        """
        response = requests.delete(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/archival/{memory_id}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete archival memory: {response.text}"

    # messages (recall memory)

    def get_messages(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Message]:
        """
        Get messages from an agent with pagination.

        Args:
            agent_id (str): ID of the agent
            before (str): Get messages before a certain time
            after (str): Get messages after a certain time
            limit (int): Limit number of messages

        Returns:
            messages (List[Message]): List of messages
        """

        params = {"before": before, "after": after, "limit": limit, "msg_object": True}
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/messages", params=params, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get messages: {response.text}")
        return [Message(**message) for message in response.json()]

    def send_message(
        self,
        agent_id: str,
        message: str,
        role: str,
        name: Optional[str] = None,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        include_full_message: Optional[bool] = False,
    ) -> Union[MemGPTResponse, Generator[MemGPTStreamingResponse, None, None]]:
        """
        Send a message to an agent

        Args:
            message (str): Message to send
            role (str): Role of the message
            agent_id (str): ID of the agent
            name(str): Name of the sender
            stream (bool): Stream the response (default: `False`)
            stream_tokens (bool): Stream tokens (default: `False`)

        Returns:
            response (MemGPTResponse): Response from the agent
        """
        # TODO: implement include_full_message
        messages = [MessageCreate(role=MessageRole(role), text=message, name=name)]
        # TODO: figure out how to handle stream_steps and stream_tokens

        # When streaming steps is True, stream_tokens must be False
        request = MemGPTRequest(messages=messages, stream_steps=stream_steps, stream_tokens=stream_tokens, return_message_object=True)
        if stream_tokens or stream_steps:
            from memgpt.client.streaming import _sse_post

            request.return_message_object = False
            return _sse_post(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/messages", request.model_dump(), self.headers)
        else:
            response = requests.post(
                f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/messages", json=request.model_dump(), headers=self.headers
            )
            if response.status_code != 200:
                raise ValueError(f"Failed to send message: {response.text}")
            response = MemGPTResponse(**response.json())

            # simplify messages
            if not include_full_message:
                messages = []
                for message in response.messages:
                    messages += message.to_memgpt_message()
                response.messages = messages

            return response

    # humans / personas

    def list_blocks(self, label: Optional[str] = None, templates_only: Optional[bool] = True) -> List[Block]:
        params = {"label": label, "templates_only": templates_only}
        response = requests.get(f"{self.base_url}/{self.api_prefix}/blocks", params=params, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list blocks: {response.text}")

        if label == "human":
            return [Human(**human) for human in response.json()]
        elif label == "persona":
            return [Persona(**persona) for persona in response.json()]
        else:
            return [Block(**block) for block in response.json()]

    def create_block(self, label: str, name: str, text: str) -> Block:  #
        request = CreateBlock(label=label, name=name, value=text)
        response = requests.post(f"{self.base_url}/{self.api_prefix}/blocks", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create block: {response.text}")
        if request.label == "human":
            return Human(**response.json())
        elif request.label == "persona":
            return Persona(**response.json())
        else:
            return Block(**response.json())

    def update_block(self, block_id: str, name: Optional[str] = None, text: Optional[str] = None) -> Block:
        request = UpdateBlock(id=block_id, name=name, value=text)
        response = requests.post(f"{self.base_url}/{self.api_prefix}/blocks/{block_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update block: {response.text}")
        return Block(**response.json())

    def get_block(self, block_id: str) -> Block:
        response = requests.get(f"{self.base_url}/{self.api_prefix}/blocks/{block_id}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get block: {response.text}")
        return Block(**response.json())

    def get_block_id(self, name: str, label: str) -> str:
        params = {"name": name, "label": label}
        response = requests.get(f"{self.base_url}/{self.api_prefix}/blocks", params=params, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get block ID: {response.text}")
        blocks = [Block(**block) for block in response.json()]
        if len(blocks) == 0:
            return None
        elif len(blocks) > 1:
            raise ValueError(f"Multiple blocks found with name {name}")
        return blocks[0].id

    def delete_block(self, id: str) -> Block:
        response = requests.delete(f"{self.base_url}/{self.api_prefix}/blocks/{id}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete block: {response.text}"
        if response.status_code != 200:
            raise ValueError(f"Failed to delete block: {response.text}")
        return Block(**response.json())

    def list_humans(self):
        """
        List available human block templates

        Returns:
            humans (List[Human]): List of human blocks
        """
        blocks = self.list_blocks(label="human")
        return [Human(**block.model_dump()) for block in blocks]

    def create_human(self, name: str, text: str) -> Human:
        """
        Create a human block template (saved human string to pre-fill `ChatMemory`)

        Args:
            name (str): Name of the human block
            text (str): Text of the human block

        Returns:
            human (Human): Human block
        """
        return self.create_block(label="human", name=name, text=text)

    def update_human(self, human_id: str, name: Optional[str] = None, text: Optional[str] = None) -> Human:
        """
        Update a human block template

        Args:
            human_id (str): ID of the human block
            text (str): Text of the human block

        Returns:
            human (Human): Updated human block
        """
        request = UpdateHuman(id=human_id, name=name, value=text)
        response = requests.post(f"{self.base_url}/{self.api_prefix}/blocks/{human_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update human: {response.text}")
        return Human(**response.json())

    def list_personas(self):
        """
        List available persona block templates

        Returns:
            personas (List[Persona]): List of persona blocks
        """
        blocks = self.list_blocks(label="persona")
        return [Persona(**block.model_dump()) for block in blocks]

    def create_persona(self, name: str, text: str) -> Persona:
        """
        Create a persona block template (saved persona string to pre-fill `ChatMemory`)

        Args:
            name (str): Name of the persona block
            text (str): Text of the persona block

        Returns:
            persona (Persona): Persona block
        """
        return self.create_block(label="persona", name=name, text=text)

    def update_persona(self, persona_id: str, name: Optional[str] = None, text: Optional[str] = None) -> Persona:
        """
        Update a persona block template

        Args:
            persona_id (str): ID of the persona block
            text (str): Text of the persona block

        Returns:
            persona (Persona): Updated persona block
        """
        request = UpdatePersona(id=persona_id, name=name, value=text)
        response = requests.post(f"{self.base_url}/{self.api_prefix}/blocks/{persona_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update persona: {response.text}")
        return Persona(**response.json())

    def get_persona(self, persona_id: str) -> Persona:
        """
        Get a persona block template

        Args:
            id (str): ID of the persona block

        Returns:
            persona (Persona): Persona block
        """
        return self.get_block(persona_id)

    def get_persona_id(self, name: str) -> str:
        """
        Get the ID of a persona block template

        Args:
            name (str): Name of the persona block

        Returns:
            id (str): ID of the persona block
        """
        return self.get_block_id(name, "persona")

    def delete_persona(self, persona_id: str) -> Persona:
        """
        Delete a persona block template

        Args:
            id (str): ID of the persona block
        """
        return self.delete_block(persona_id)

    def get_human(self, human_id: str) -> Human:
        """
        Get a human block template

        Args:
            id (str): ID of the human block

        Returns:
            human (Human): Human block
        """
        return self.get_block(human_id)

    def get_human_id(self, name: str) -> str:
        """
        Get the ID of a human block template

        Args:
            name (str): Name of the human block

        Returns:
            id (str): ID of the human block
        """
        return self.get_block_id(name, "human")

    def delete_human(self, human_id: str) -> Human:
        """
        Delete a human block template

        Args:
            id (str): ID of the human block
        """
        return self.delete_block(human_id)

    # sources

    def get_source(self, source_id: str) -> Source:
        """
        Get a source given the ID.

        Args:
            source_id (str): ID of the source

        Returns:
            source (Source): Source
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/sources/{source_id}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get source: {response.text}")
        return Source(**response.json())

    def get_source_id(self, source_name: str) -> str:
        """
        Get the ID of a source

        Args:
            source_name (str): Name of the source

        Returns:
            source_id (str): ID of the source
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/sources/name/{source_name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get source ID: {response.text}")
        return response.json()

    def list_sources(self) -> List[Source]:
        """
        List available sources

        Returns:
            sources (List[Source]): List of sources
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/sources", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list sources: {response.text}")
        return [Source(**source) for source in response.json()]

    def delete_source(self, source_id: str):
        """
        Delete a source

        Args:
            source_id (str): ID of the source
        """
        response = requests.delete(f"{self.base_url}/{self.api_prefix}/sources/{str(source_id)}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete source: {response.text}"

    def get_job(self, job_id: str) -> Job:
        response = requests.get(f"{self.base_url}/{self.api_prefix}/jobs/{job_id}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get job: {response.text}")
        return Job(**response.json())

    def list_jobs(self):
        response = requests.get(f"{self.base_url}/{self.api_prefix}/jobs", headers=self.headers)
        return [Job(**job) for job in response.json()]

    def list_active_jobs(self):
        response = requests.get(f"{self.base_url}/{self.api_prefix}/jobs/active", headers=self.headers)
        return [Job(**job) for job in response.json()]

    def load_data(self, connector: DataConnector, source_name: str):
        raise NotImplementedError

    def load_file_into_source(self, filename: str, source_id: str, blocking=True):
        """
        Load a file into a source

        Args:
            filename (str): Name of the file
            source_id (str): ID of the source
            blocking (bool): Block until the job is complete

        Returns:
            job (Job): Data loading job including job status and metadata
        """
        files = {"file": open(filename, "rb")}

        # create job
        response = requests.post(f"{self.base_url}/{self.api_prefix}/sources/{source_id}/upload", files=files, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to upload file to source: {response.text}")

        job = Job(**response.json())
        if blocking:
            # wait until job is completed
            while True:
                job = self.get_job(job.id)
                if job.status == JobStatus.completed:
                    break
                elif job.status == JobStatus.failed:
                    raise ValueError(f"Job failed: {job.metadata}")
                time.sleep(1)
        return job

    def create_source(self, name: str) -> Source:
        """
        Create a source

        Args:
            name (str): Name of the source

        Returns:
            source (Source): Created source
        """
        payload = {"name": name}
        response = requests.post(f"{self.base_url}/{self.api_prefix}/sources", json=payload, headers=self.headers)
        response_json = response.json()
        return Source(**response_json)

    def list_attached_sources(self, agent_id: str) -> List[Source]:
        """
        List sources attached to an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            sources (List[Source]): List of sources
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/agents/{agent_id}/sources", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list attached sources: {response.text}")
        return [Source(**source) for source in response.json()]

    def update_source(self, source_id: str, name: Optional[str] = None) -> Source:
        """
        Update a source

        Args:
            source_id (str): ID of the source
            name (str): Name of the source

        Returns:
            source (Source): Updated source
        """
        request = SourceUpdate(id=source_id, name=name)
        response = requests.patch(f"{self.base_url}/{self.api_prefix}/sources/{source_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update source: {response.text}")
        return Source(**response.json())

    def attach_source_to_agent(self, source_id: str, agent_id: str):
        """
        Attach a source to an agent

        Args:
            agent_id (str): ID of the agent
            source_id (str): ID of the source
            source_name (str): Name of the source
        """
        params = {"agent_id": agent_id}
        response = requests.post(f"{self.base_url}/{self.api_prefix}/sources/{source_id}/attach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to attach source to agent: {response.text}"

    def detach_source(self, source_id: str, agent_id: str):
        """Detach a source from an agent"""
        params = {"agent_id": str(agent_id)}
        response = requests.post(f"{self.base_url}/{self.api_prefix}/sources/{source_id}/detach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to detach source from agent: {response.text}"

    # server configuration commands

    def list_models(self):
        """
        List available LLM models

        Returns:
            models (List[LLMConfig]): List of LLM models
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/models", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list models: {response.text}")
        return [LLMConfig(**model) for model in response.json()]

    def list_embedding_models(self):
        """
        List available embedding models

        Returns:
            models (List[EmbeddingConfig]): List of embedding models
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/models/embedding", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list embedding models: {response.text}")
        return [EmbeddingConfig(**model) for model in response.json()]

    # tools

    def get_tool_id(self, tool_name: str):
        """
        Get the ID of a tool

        Args:
            name (str): Name of the tool

        Returns:
            id (str): ID of the tool (`None` if not found)
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/tools/name/{tool_name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return response.json()

    def create_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        update: Optional[bool] = True,  # TODO: actually use this
        tags: Optional[List[str]] = None,
    ) -> Tool:
        """
        Create a tool. This stores the source code of function on the server, so that the server can execute the function and generate an OpenAI JSON schemas for it when using with an agent.

        Args:
            func (callable): The function to create a tool for.
            name: (str): Name of the tool (must be unique per-user.)
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            update (bool, optional): Update the tool if it already exists. Defaults to True.

        Returns:
            tool (Tool): The created tool.
        """

        # TODO: check tool update code
        # TODO: check if tool already exists

        # TODO: how to load modules?
        # parse source code/schema
        source_code = parse_source_code(func)
        source_type = "python"

        # TODO: Check if tool already exists
        # if name:
        #     tool_id = self.get_tool_id(tool_name=name)
        #     if tool_id:
        #         raise ValueError(f"Tool with name {name} (id={tool_id}) already exists")

        # call server function
        request = ToolCreate(source_type=source_type, source_code=source_code, name=name, tags=tags)
        response = requests.post(f"{self.base_url}/{self.api_prefix}/tools", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create tool: {response.text}")
        return Tool(**response.json())

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        func: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
    ) -> Tool:
        """
        Update a tool with provided parameters (name, func, tags)

        Args:
            id (str): ID of the tool
            name (str): Name of the tool
            func (callable): Function to wrap in a tool
            tags (List[str]): Tags for the tool

        Returns:
            tool (Tool): Updated tool
        """
        if func:
            source_code = parse_source_code(func)
        else:
            source_code = None

        source_type = "python"

        request = ToolUpdate(id=id, source_type=source_type, source_code=source_code, tags=tags, name=name)
        response = requests.patch(f"{self.base_url}/{self.api_prefix}/tools/{id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update tool: {response.text}")
        return Tool(**response.json())

    # def create_tool(
    #    self,
    #    func,
    #    name: Optional[str] = None,
    #    update: Optional[bool] = True,  # TODO: actually use this
    #    tags: Optional[List[str]] = None,
    # ):
    #    """Create a tool

    #    Args:
    #        func (callable): The function to create a tool for.
    #        tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
    #        update (bool, optional): Update the tool if it already exists. Defaults to True.

    #    Returns:
    #        Tool object
    #    """

    #    # TODO: check if tool already exists
    #    # TODO: how to load modules?
    #    # parse source code/schema
    #    source_code = parse_source_code(func)
    #    json_schema = generate_schema(func, name)
    #    source_type = "python"
    #    json_schema["name"]

    #    # create data
    #    data = {"source_code": source_code, "source_type": source_type, "tags": tags, "json_schema": json_schema, "update": update}
    #    try:
    #        CreateToolRequest(**data)  # validate data
    #    except Exception as e:
    #        raise ValueError(f"Failed to create tool: {e}, invalid input {data}")

    #    # make REST request
    #    response = requests.post(f"{self.base_url}/{self.api_prefix}/tools", json=data, headers=self.headers)
    #    if response.status_code != 200:
    #        raise ValueError(f"Failed to create tool: {response.text}")
    #    return ToolModel(**response.json())

    def list_tools(self) -> List[Tool]:
        """
        List available tools for the user.

        Returns:
            tools (List[Tool]): List of tools
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/tools", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list tools: {response.text}")
        return [Tool(**tool) for tool in response.json()]

    def delete_tool(self, name: str):
        """
        Delete a tool given the ID.

        Args:
            id (str): ID of the tool
        """
        response = requests.delete(f"{self.base_url}/{self.api_prefix}/tools/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to delete tool: {response.text}")

    def get_tool(self, id: str) -> Optional[Tool]:
        """
        Get a tool give its ID.

        Args:
            id (str): ID of the tool

        Returns:
            tool (Tool): Tool
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/tools/{id}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return Tool(**response.json())

    def get_tool_id(self, name: str) -> Optional[str]:
        """
        Get a tool ID by its name.

        Args:
            id (str): ID of the tool

        Returns:
            tool (Tool): Tool
        """
        response = requests.get(f"{self.base_url}/{self.api_prefix}/tools/name/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return response.json()


class LocalClient(AbstractClient):
    """
    A local client for MemGPT, which corresponds to a single user.

    Attributes:
        auto_save (bool): Whether to automatically save changes.
        user_id (str): The user ID.
        debug (bool): Whether to print debug information.
        interface (QueuingInterface): The interface for the client.
        server (SyncServer): The server for the client.
    """

    def __init__(
        self,
        auto_save: bool = False,
        user_id: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.

        Args:
            auto_save (bool): Whether to automatically save changes.
            user_id (str): The user ID.
            debug (bool): Whether to print debug information.
        """
        self.auto_save = auto_save

        # determine user_id (pulled from local config)
        config = MemGPTConfig.load()
        if user_id:
            self.user_id = user_id
        else:
            # TODO: find a neater way to do this
            self.user_id = config.anon_clientid

        # set logging levels
        memgpt.utils.DEBUG = debug
        logging.getLogger().setLevel(logging.CRITICAL)

        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface_factory=lambda: self.interface)

        # set logging levels
        memgpt.utils.DEBUG = debug
        logging.getLogger().setLevel(logging.CRITICAL)

        # create user if does not exist
        existing_user = self.server.get_user(self.user_id)
        if not existing_user:
            self.user = self.server.create_user(UserCreate())
            self.user_id = self.user.id

            # update config
            config.anon_clientid = str(self.user_id)
            config.save()

    # agents

    def list_agents(self) -> List[AgentState]:
        self.interface.clear()

        # TODO: fix the server function
        # return self.server.list_agents(user_id=self.user_id)

        return self.server.ms.list_agents(user_id=self.user_id)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """
        Check if an agent exists

        Args:
            agent_id (str): ID of the agent
            agent_name (str): Name of the agent

        Returns:
            exists (bool): `True` if the agent exists, `False` otherwise
        """

        if not (agent_id or agent_name):
            raise ValueError(f"Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError(f"Only one of agent_id or agent_name can be provided")
        existing = self.list_agents()
        if agent_id:
            return str(agent_id) in [str(agent.id) for agent in existing]
        else:
            return agent_name in [str(agent.name) for agent in existing]

    def create_agent(
        self,
        name: Optional[str] = None,
        # model configs
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        # memory
        memory: Memory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        # system
        system: Optional[str] = None,
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        # metadata
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
        description: Optional[str] = None,
    ) -> AgentState:
        """Create an agent

        Args:
            name (str): Name of the agent
            embedding_config (EmbeddingConfig): Embedding configuration
            llm_config (LLMConfig): LLM configuration
            memory (Memory): Memory configuration
            system (str): System configuration
            tools (List[str]): List of tools
            include_base_tools (bool): Include base tools
            metadata (Dict): Metadata
            description (str): Description

        Returns:
            agent_state (AgentState): State of the created agent
        """

        if name and self.agent_exists(agent_name=name):
            raise ValueError(f"Agent with name {name} already exists (user_id={self.user_id})")

        # construct list of tools
        tool_names = []
        if tools:
            tool_names += tools
        if include_base_tools:
            tool_names += BASE_TOOLS

        # add memory tools
        memory_functions = get_memory_functions(memory)
        for func_name, func in memory_functions.items():
            tool = self.create_tool(func, name=func_name, tags=["memory", "memgpt-base"], update=True)
            tool_names.append(tool.name)

        self.interface.clear()

        # create agent
        agent_state = self.server.create_agent(
            CreateAgent(
                name=name,
                description=description,
                metadata_=metadata,
                memory=memory,
                tools=tool_names,
                system=system,
                llm_config=llm_config,
                embedding_config=embedding_config,
            ),
            user_id=self.user_id,
        )
        return agent_state

    def update_message(
        self,
        agent_id: str,
        message_id: str,
        role: Optional[MessageRole] = None,
        text: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        message = self.server.update_agent_message(
            agent_id=agent_id,
            request=UpdateMessage(
                id=message_id,
                role=role,
                text=text,
                name=name,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            ),
        )
        return message

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        message_ids: Optional[List[str]] = None,
        memory: Optional[Memory] = None,
    ):
        """
        Update an existing agent

        Args:
            agent_id (str): ID of the agent
            name (str): Name of the agent
            description (str): Description of the agent
            system (str): System configuration
            tools (List[str]): List of tools
            metadata (Dict): Metadata
            llm_config (LLMConfig): LLM configuration
            embedding_config (EmbeddingConfig): Embedding configuration
            message_ids (List[str]): List of message IDs
            memory (Memory): Memory configuration

        Returns:
            agent_state (AgentState): State of the updated agent
        """
        self.interface.clear()
        agent_state = self.server.update_agent(
            UpdateAgentState(
                id=agent_id,
                name=name,
                system=system,
                tools=tools,
                description=description,
                metadata_=metadata,
                llm_config=llm_config,
                embedding_config=embedding_config,
                message_ids=message_ids,
                memory=memory,
            ),
            user_id=self.user_id,
        )
        return agent_state

    def rename_agent(self, agent_id: str, new_name: str):
        """
        Rename an agent

        Args:
            agent_id (str): ID of the agent
            new_name (str): New name for the agent
        """
        self.update_agent(agent_id, name=new_name)

    def delete_agent(self, agent_id: str):
        """
        Delete an agent

        Args:
            agent_id (str): ID of the agent to delete
        """
        self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)

    def get_agent(self, agent_id: str) -> AgentState:
        """
        Get an agent's state by it's ID.

        Args:
            agent_id (str): ID of the agent

        Returns:
            agent_state (AgentState): State representation of the agent
        """
        # TODO: include agent_name
        self.interface.clear()
        return self.server.get_agent_state(user_id=self.user_id, agent_id=agent_id)

    def get_agent_id(self, agent_name: str) -> AgentState:
        """
        Get the ID of an agent by name (names are unique per user)

        Args:
            agent_name (str): Name of the agent

        Returns:
            agent_id (str): ID of the agent
        """

        self.interface.clear()
        assert agent_name, f"Agent name must be provided"
        return self.server.get_agent_id(name=agent_name, user_id=self.user_id)

    # memory
    def get_in_context_memory(self, agent_id: str) -> Memory:
        """
        Get the in-contxt (i.e. core) memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            memory (Memory): In-context memory of the agent
        """
        memory = self.server.get_agent_memory(agent_id=agent_id)
        return memory

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        """
        Update the in-context memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            memory (Memory): The updated in-context memory of the agent

        """
        # TODO: implement this (not sure what it should look like)
        memory = self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents={section: value})
        return memory

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        """
        Get a summary of the archival memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            summary (ArchivalMemorySummary): Summary of the archival memory

        """
        return self.server.get_archival_memory_summary(agent_id=agent_id)

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        """
        Get a summary of the recall memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            summary (RecallMemorySummary): Summary of the recall memory
        """
        return self.server.get_recall_memory_summary(agent_id=agent_id)

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """
        Get in-context messages of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            messages (List[Message]): List of in-context messages
        """
        return self.server.get_in_context_messages(agent_id=agent_id)

    # agent interactions

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        include_full_message: Optional[bool] = False,
    ) -> MemGPTResponse:
        """
        Send a message to an agent

        Args:
            message (str): Message to send
            role (str): Role of the message
            agent_id (str): ID of the agent
            name(str): Name of the sender
            stream (bool): Stream the response (default: `False`)

        Returns:
            response (MemGPTResponse): Response from the agent
        """
        if not agent_id:
            assert agent_name, f"Either agent_id or agent_name must be provided"
            raise NotImplementedError
            # agent_state = self.get_agent(agent_name=agent_name)
            # agent_id = agent_state.id
        agent_state = self.get_agent(agent_id=agent_id)

        if stream_steps or stream_tokens:
            # TODO: implement streaming with stream=True/False
            raise NotImplementedError
        self.interface.clear()
        if role == "system":
            usage = self.server.system_message(user_id=self.user_id, agent_id=agent_id, message=message)
        elif role == "user":
            usage = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        else:
            raise ValueError(f"Role {role} not supported")

        # auto-save
        if self.auto_save:
            self.save()

        # TODO: need to make sure date/timestamp is propely passed
        # TODO: update self.interface.to_list() to return actual Message objects
        #       here, the message objects will have faulty created_by timestamps
        messages = self.interface.to_list()
        for m in messages:
            assert isinstance(m, Message), f"Expected Message object, got {type(m)}"
        memgpt_messages = []
        for m in messages:
            memgpt_messages += m.to_memgpt_message()
        return MemGPTResponse(messages=memgpt_messages, usage=usage)

        # format messages
        if include_full_message:
            memgpt_messages = messages
        else:
            memgpt_messages = []
            for m in messages:
                memgpt_messages += m.to_memgpt_message()

        return MemGPTResponse(messages=memgpt_messages, usage=usage)

    def user_message(self, agent_id: str, message: str, include_full_message: Optional[bool] = False) -> MemGPTResponse:
        """
        Send a message to an agent as a user

        Args:
            agent_id (str): ID of the agent
            message (str): Message to send

        Returns:
            response (MemGPTResponse): Response from the agent
        """
        self.interface.clear()
        return self.send_message(role="user", agent_id=agent_id, message=message, include_full_message=include_full_message)

    def run_command(self, agent_id: str, command: str) -> MemGPTResponse:
        """
        Run a command on the agent

        Args:
            agent_id (str): The agent ID
            command (str): The command to run

        Returns:
            MemGPTResponse: The response from the agent

        """
        self.interface.clear()
        usage = self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

        # auto-save
        if self.auto_save:
            self.save()

        # NOTE: messages/usage may be empty, depending on the command
        return MemGPTResponse(messages=self.interface.to_list(), usage=usage)

    def save(self):
        self.server.save_agents()

    # archival memory

    # humans / personas

    def create_human(self, name: str, text: str):
        """
        Create a human block template (saved human string to pre-fill `ChatMemory`)

        Args:
            name (str): Name of the human block
            text (str): Text of the human block

        Returns:
            human (Human): Human block
        """
        return self.server.create_block(CreateHuman(name=name, value=text, user_id=self.user_id), user_id=self.user_id)

    def create_persona(self, name: str, text: str):
        """
        Create a persona block template (saved persona string to pre-fill `ChatMemory`)

        Args:
            name (str): Name of the persona block
            text (str): Text of the persona block

        Returns:
            persona (Persona): Persona block
        """
        return self.server.create_block(CreatePersona(name=name, value=text, user_id=self.user_id), user_id=self.user_id)

    def list_humans(self):
        """
        List available human block templates

        Returns:
            humans (List[Human]): List of human blocks
        """
        return self.server.get_blocks(label="human", user_id=self.user_id, template=True)

    def list_personas(self) -> List[Persona]:
        """
        List available persona block templates

        Returns:
            personas (List[Persona]): List of persona blocks
        """
        return self.server.get_blocks(label="persona", user_id=self.user_id, template=True)

    def update_human(self, human_id: str, text: str):
        """
        Update a human block template

        Args:
            human_id (str): ID of the human block
            text (str): Text of the human block

        Returns:
            human (Human): Updated human block
        """
        return self.server.update_block(UpdateHuman(id=human_id, value=text, user_id=self.user_id, template=True))

    def update_persona(self, persona_id: str, text: str):
        """
        Update a persona block template

        Args:
            persona_id (str): ID of the persona block
            text (str): Text of the persona block

        Returns:
            persona (Persona): Updated persona block
        """
        return self.server.update_block(UpdatePersona(id=persona_id, value=text, user_id=self.user_id, template=True))

    def get_persona(self, id: str) -> Persona:
        """
        Get a persona block template

        Args:
            id (str): ID of the persona block

        Returns:
            persona (Persona): Persona block
        """
        assert id, f"Persona ID must be provided"
        return Persona(**self.server.get_block(id).model_dump())

    def get_human(self, id: str) -> Human:
        """
        Get a human block template

        Args:
            id (str): ID of the human block

        Returns:
            human (Human): Human block
        """
        assert id, f"Human ID must be provided"
        return Human(**self.server.get_block(id).model_dump())

    def get_persona_id(self, name: str) -> str:
        """
        Get the ID of a persona block template

        Args:
            name (str): Name of the persona block

        Returns:
            id (str): ID of the persona block
        """
        persona = self.server.get_blocks(name=name, label="persona", user_id=self.user_id, template=True)
        if not persona:
            return None
        return persona[0].id

    def get_human_id(self, name: str) -> str:
        """
        Get the ID of a human block template

        Args:
            name (str): Name of the human block

        Returns:
            id (str): ID of the human block
        """
        human = self.server.get_blocks(name=name, label="human", user_id=self.user_id, template=True)
        if not human:
            return None
        return human[0].id

    def delete_persona(self, id: str):
        """
        Delete a persona block template

        Args:
            id (str): ID of the persona block
        """
        self.server.delete_block(id)

    def delete_human(self, id: str):
        """
        Delete a human block template

        Args:
            id (str): ID of the human block
        """
        self.server.delete_block(id)

    # tools

    # TODO: merge this into create_tool
    def add_tool(self, tool: Tool, update: Optional[bool] = True) -> Tool:
        """
        Adds a tool directly.

        Args:
            tool (Tool): The tool to add.
            update (bool, optional): Update the tool if it already exists. Defaults to True.

        Returns:
            None
        """
        existing_tool_id = self.get_tool_id(tool.name)
        if existing_tool_id:
            if update:
                self.server.update_tool(
                    ToolUpdate(
                        id=existing_tool_id,
                        source_type=tool.source_type,
                        source_code=tool.source_code,
                        tags=tool.tags,
                        json_schema=tool.json_schema,
                        name=tool.name,
                    )
                )
            else:
                raise ValueError(f"Tool with name {tool.name} already exists")

        # call server function
        return self.server.create_tool(
            ToolCreate(
                source_type=tool.source_type, source_code=tool.source_code, name=tool.name, json_schema=tool.json_schema, tags=tool.tags
            ),
            user_id=self.user_id,
            update=update,
        )

    # TODO: Use the above function `add_tool` here as there is duplicate logic
    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        update: Optional[bool] = True,  # TODO: actually use this
        tags: Optional[List[str]] = None,
    ) -> Tool:
        """
        Create a tool. This stores the source code of function on the server, so that the server can execute the function and generate an OpenAI JSON schemas for it when using with an agent.

        Args:
            func (callable): The function to create a tool for.
            name: (str): Name of the tool (must be unique per-user.)
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            update (bool, optional): Update the tool if it already exists. Defaults to True.

        Returns:
            tool (Tool): The created tool.
        """
        # TODO: check if tool already exists
        # TODO: how to load modules?
        # parse source code/schema
        source_code = parse_source_code(func)
        source_type = "python"
        if not tags:
            tags = []

        # call server function
        return self.server.create_tool(
            # ToolCreate(source_type=source_type, source_code=source_code, name=tool_name, json_schema=json_schema, tags=tags),
            ToolCreate(source_type=source_type, source_code=source_code, name=name, tags=tags),
            user_id=self.user_id,
            update=update,
        )

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        func: Optional[callable] = None,
        tags: Optional[List[str]] = None,
    ) -> Tool:
        """
        Update a tool with provided parameters (name, func, tags)

        Args:
            id (str): ID of the tool
            name (str): Name of the tool
            func (callable): Function to wrap in a tool
            tags (List[str]): Tags for the tool

        Returns:
            tool (Tool): Updated tool
        """
        if func:
            source_code = parse_source_code(func)
        else:
            source_code = None

        source_type = "python"

        return self.server.update_tool(ToolUpdate(id=id, source_type=source_type, source_code=source_code, tags=tags, name=name))

    def list_tools(self):
        """
        List available tools for the user.

        Returns:
            tools (List[Tool]): List of tools
        """
        tools = self.server.list_tools(user_id=self.user_id)
        return tools

    def get_tool(self, id: str) -> Optional[Tool]:
        """
        Get a tool give its ID.

        Args:
            id (str): ID of the tool

        Returns:
            tool (Tool): Tool
        """
        return self.server.get_tool(id)

    def delete_tool(self, id: str):
        """
        Delete a tool given the ID.

        Args:
            id (str): ID of the tool
        """
        return self.server.delete_tool(id)

    def get_tool_id(self, name: str) -> Optional[str]:
        """
        Get the ID of a tool

        Args:
            name (str): Name of the tool

        Returns:
            id (str): ID of the tool (`None` if not found)
        """
        return self.server.get_tool_id(name, self.user_id)

    # data sources

    def load_data(self, connector: DataConnector, source_name: str):
        """
        Load data into a source

        Args:
            connector (DataConnector): Data connector
            source_name (str): Name of the source
        """
        self.server.load_data(user_id=self.user_id, connector=connector, source_name=source_name)

    def load_file_into_source(self, filename: str, source_id: str, blocking=True):
        """
        Load a file into a source

        Args:
            filename (str): Name of the file
            source_id (str): ID of the source
            blocking (bool): Block until the job is complete

        Returns:
            job (Job): Data loading job including job status and metadata
        """
        job = self.server.create_job(user_id=self.user_id)

        # TODO: implement blocking vs. non-blocking
        self.server.load_file_to_source(source_id=source_id, file_path=filename, job_id=job.id)
        return job

    def get_job(self, job_id: str):
        return self.server.get_job(job_id=job_id)

    def list_jobs(self):
        return self.server.list_jobs(user_id=self.user_id)

    def list_active_jobs(self):
        return self.server.list_active_jobs(user_id=self.user_id)

    def create_source(self, name: str) -> Source:
        """
        Create a source

        Args:
            name (str): Name of the source

        Returns:
            source (Source): Created source
        """
        request = SourceCreate(name=name)
        return self.server.create_source(request=request, user_id=self.user_id)

    def delete_source(self, source_id: str):
        """
        Delete a source

        Args:
            source_id (str): ID of the source
        """

        # TODO: delete source data
        self.server.delete_source(source_id=source_id, user_id=self.user_id)

    def get_source(self, source_id: str) -> Source:
        """
        Get a source given the ID.

        Args:
            source_id (str): ID of the source

        Returns:
            source (Source): Source
        """
        return self.server.get_source(source_id=source_id, user_id=self.user_id)

    def get_source_id(self, source_name: str) -> str:
        """
        Get the ID of a source

        Args:
            source_name (str): Name of the source

        Returns:
            source_id (str): ID of the source
        """
        return self.server.get_source_id(source_name=source_name, user_id=self.user_id)

    def attach_source_to_agent(self, agent_id: str, source_id: Optional[str] = None, source_name: Optional[str] = None):
        """
        Attach a source to an agent

        Args:
            agent_id (str): ID of the agent
            source_id (str): ID of the source
            source_name (str): Name of the source
        """
        self.server.attach_source_to_agent(source_id=source_id, source_name=source_name, agent_id=agent_id, user_id=self.user_id)

    def detach_source_from_agent(self, agent_id: str, source_id: Optional[str] = None, source_name: Optional[str] = None):
        self.server.detach_source_from_agent(source_id=source_id, source_name=source_name, agent_id=agent_id, user_id=self.user_id)

    def list_sources(self) -> List[Source]:
        """
        List available sources

        Returns:
            sources (List[Source]): List of sources
        """

        return self.server.list_all_sources(user_id=self.user_id)

    def list_attached_sources(self, agent_id: str) -> List[Source]:
        """
        List sources attached to an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            sources (List[Source]): List of sources
        """
        return self.server.list_attached_sources(agent_id=agent_id)

    def update_source(self, source_id: str, name: Optional[str] = None) -> Source:
        """
        Update a source

        Args:
            source_id (str): ID of the source
            name (str): Name of the source

        Returns:
            source (Source): Updated source
        """
        # TODO should the arg here just be "source_update: Source"?
        request = SourceUpdate(id=source_id, name=name)
        return self.server.update_source(request=request, user_id=self.user_id)

    # archival memory

    def insert_archival_memory(self, agent_id: str, memory: str) -> List[Passage]:
        """
        Insert archival memory into an agent

        Args:
            agent_id (str): ID of the agent
            memory (str): Memory string to insert

        Returns:
            passages (List[Passage]): List of inserted passages
        """
        return self.server.insert_archival_memory(user_id=self.user_id, agent_id=agent_id, memory_contents=memory)

    def delete_archival_memory(self, agent_id: str, memory_id: str):
        """
        Delete archival memory from an agent

        Args:
            agent_id (str): ID of the agent
            memory_id (str): ID of the memory
        """
        self.server.delete_archival_memory(user_id=self.user_id, agent_id=agent_id, memory_id=memory_id)

    def get_archival_memory(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Passage]:
        """
        Get archival memory from an agent with pagination.

        Args:
            agent_id (str): ID of the agent
            before (str): Get memories before a certain time
            after (str): Get memories after a certain time
            limit (int): Limit number of memories

        Returns:
            passages (List[Passage]): List of passages
        """

        return self.server.get_agent_archival_cursor(user_id=self.user_id, agent_id=agent_id, before=before, after=after, limit=limit)

    # recall memory

    def get_messages(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Message]:
        """
        Get messages from an agent with pagination.

        Args:
            agent_id (str): ID of the agent
            before (str): Get messages before a certain time
            after (str): Get messages after a certain time
            limit (int): Limit number of messages

        Returns:
            messages (List[Message]): List of messages
        """

        self.interface.clear()
        return self.server.get_agent_recall_cursor(
            user_id=self.user_id,
            agent_id=agent_id,
            before=before,
            after=after,
            limit=limit,
            reverse=True,
            return_message_object=True,
        )

    def list_models(self) -> List[LLMConfig]:
        """
        List available LLM models

        Returns:
            models (List[LLMConfig]): List of LLM models
        """
        return self.server.list_models()

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        """
        List available embedding models

        Returns:
            models (List[EmbeddingConfig]): List of embedding models
        """
        return [self.server.server_embedding_config]

    def list_blocks(self, label: Optional[str] = None, templates_only: Optional[bool] = True) -> List[Block]:
        """
        List available blocks

        Args:
            label (str): Label of the block
            templates_only (bool): List only templates

        Returns:
            blocks (List[Block]): List of blocks
        """
        return self.server.get_blocks(label=label, template=templates_only)

    def create_block(self, name: str, text: str, label: Optional[str] = None) -> Block:  #
        """
        Create a block

        Args:
            label (str): Label of the block
            name (str): Name of the block
            text (str): Text of the block

        Returns:
            block (Block): Created block
        """
        return self.server.create_block(CreateBlock(label=label, name=name, value=text, user_id=self.user_id), user_id=self.user_id)

    def update_block(self, block_id: str, name: Optional[str] = None, text: Optional[str] = None) -> Block:
        """
        Update a block

        Args:
            block_id (str): ID of the block
            name (str): Name of the block
            text (str): Text of the block

        Returns:
            block (Block): Updated block
        """
        return self.server.update_block(UpdateBlock(id=block_id, name=name, value=text))

    def get_block(self, block_id: str) -> Block:
        """
        Get a block

        Args:
            block_id (str): ID of the block

        Returns:
            block (Block): Block
        """
        return self.server.get_block(block_id)

    def delete_block(self, id: str) -> Block:
        """
        Delete a block

        Args:
            id (str): ID of the block

        Returns:
            block (Block): Deleted block
        """
        return self.server.delete_block(id)
