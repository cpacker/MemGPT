import datetime
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

import requests

from memgpt.agent import Agent
from memgpt.config import MemGPTConfig
from memgpt.constants import BASE_TOOLS, DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.data_types import AgentState, EmbeddingConfig, LLMConfig, Source
from memgpt.functions.functions import parse_source_code
from memgpt.functions.schema_generator import generate_schema
from memgpt.memory import BaseMemory, ChatMemory, get_memory_functions
from memgpt.models.pydantic_models import (
    AgentStateModel,
    EmbeddingConfigModel,
    HumanModel,
    JobModel,
    JobStatus,
    LLMConfigModel,
    PersonaModel,
    SourceModel,
    ToolModel,
    UserModel,
)
from memgpt.server.rest_api.agents.command import CommandResponse
from memgpt.server.rest_api.agents.config import GetAgentResponse
from memgpt.server.rest_api.agents.index import CreateAgentResponse, ListAgentsResponse
from memgpt.server.rest_api.agents.memory import (
    ArchivalMemoryObject,
    GetAgentArchivalMemoryResponse,
    GetAgentMemoryResponse,
    InsertAgentArchivalMemoryResponse,
    UpdateAgentMemoryResponse,
)
from memgpt.server.rest_api.agents.message import (
    GetAgentMessagesResponse,
    UserMessageResponse,
)
from memgpt.server.rest_api.config.index import ConfigResponse
from memgpt.server.rest_api.humans.index import ListHumansResponse
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.models.index import ListModelsResponse
from memgpt.server.rest_api.personas.index import ListPersonasResponse
from memgpt.server.rest_api.sources.index import ListSourcesResponse

# import pydantic response objects from memgpt.server.rest_api
from memgpt.server.rest_api.tools.index import CreateToolRequest, ListToolsResponse
from memgpt.server.server import SyncServer
from memgpt.streaming_interface import (
    StreamingRefreshCLIInterface as interface,  # for printing to terminal
)
from memgpt.utils import get_human_text


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

    # agents

    def list_agents(self):
        """List all agents associated with a given user."""
        raise NotImplementedError

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """Check if an agent with the specified ID or name exists."""
        raise NotImplementedError

    def create_agent(
        self,
        name: Optional[str] = None,
        preset: Optional[str] = None,
        persona: Optional[str] = None,
        human: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        """Create a new agent with the specified configuration."""
        raise NotImplementedError

    def rename_agent(self, agent_id: uuid.UUID, new_name: str):
        """Rename the agent."""
        raise NotImplementedError

    def delete_agent(self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None):
        """Delete the agent."""
        raise NotImplementedError

    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        raise NotImplementedError

    def get_agent_config(self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None) -> AgentState:
        raise NotImplementedError

    def update_agent(agent_state: AgentState):
        raise NotImplementedError

    def save_agent(self, agent: Agent):
        raise NotImplementedError

    # users
    def get_user(self) -> Optional[UserModel]:
        """Gets user"""
        raise NotImplementedError

    # memory

    def get_agent_memory(self, agent_id: str) -> Dict:
        raise NotImplementedError

    def update_agent_core_memory(self, agent_id: str, human: Optional[str] = None, persona: Optional[str] = None) -> Dict:
        raise NotImplementedError

    # agent interactions

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        raise NotImplementedError

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    # archival memory

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Paginated get for the archival memory for an agent"""
        raise NotImplementedError

    def insert_archival_memory(self, agent_id: uuid.UUID, memory: str):
        """Insert archival memory into the agent."""
        raise NotImplementedError

    def delete_archival_memory(self, agent_id: uuid.UUID, memory_id: uuid.UUID):
        """Delete archival memory from the agent."""
        raise NotImplementedError

    # messages (recall memory)

    def get_messages(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Get messages for the agent."""
        raise NotImplementedError

    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False):
        """Send a message to the agent."""
        raise NotImplementedError

    # humans / personas

    def list_humans(self):
        """List all humans."""
        raise NotImplementedError

    def create_human(self, name: str, human: str):
        """Create a human."""
        raise NotImplementedError

    def get_human(self, name: str, user_id: uuid.UUID):
        """Get a human."""
        raise NotImplementedError

    def add_human(self, name: str, text: str) -> HumanModel:
        """Add new human or change existing one."""
        raise NotImplementedError

    def update_human(self, name: str, text: str) -> HumanModel:
        """Update an existing human"""
        raise NotImplementedError

    def delete_human(self, name: str, user_id: uuid.UUID):
        """Delete an existing human"""
        raise NotImplementedError

    def list_personas(self):
        """List all personas."""
        raise NotImplementedError

    def create_persona(self, name: str, persona: str):
        """Create a persona."""
        raise NotImplementedError

    def add_persona(self, user_id: uuid.UUID, name: str, persona: str, text: Optional[str]):
        """Update or create a persona."""
        raise NotImplementedError

    # tools

    def list_tools(self):
        """List all tools."""
        raise NotImplementedError

    # data sources

    def list_sources(self):
        """List loaded sources"""
        raise NotImplementedError

    def delete_source(self):
        """Delete a source and associated data (including attached to agents)"""
        raise NotImplementedError

    def load_file_into_source(self, filename: str, source_id: uuid.UUID):
        """Load {filename} and insert into source"""
        raise NotImplementedError

    def create_source(self, name: str):
        """Create a new source"""
        raise NotImplementedError

    def attach_source_to_agent(self, agent_id: uuid.UUID, source_id: Optional[uuid.UUID], source_name: Optional[str]):
        """Attach a source to an agent"""
        raise NotImplementedError

    def detach_source(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Detach a source from an agent"""
        raise NotImplementedError

    # server configuration commands

    def list_models(self):
        """List all models."""
        raise NotImplementedError

    def get_config(self):
        """Get server config"""
        raise NotImplementedError


class AgentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return {key: self.default(value) for key, value in obj.__dict__.items()}
        return super().default(obj)


class RESTClient(AbstractClient):
    def __init__(
        self,
        base_url: str,
        token: str,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.base_url = base_url
        self.headers = {"accept": "application/json", "authorization": f"Bearer {token}"}

    # agents
    def list_agents(self):
        response = requests.get(f"{self.base_url}/api/agents", headers=self.headers)
        return ListAgentsResponse(**response.json()).agents

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        response = requests.get(f"{self.base_url}/api/agents/{str(agent_id)}/config", headers=self.headers)
        if response.status_code == 404:
            # not found error
            return False
        elif response.status_code == 200:
            return True
        else:
            raise ValueError(f"Failed to check if agent exists: {response.text}")

    def get_tool(self, tool_name: str):
        response = requests.get(f"{self.base_url}/api/tools/{tool_name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return ToolModel(**response.json())

    def create_agent(
        self,
        name: Optional[str] = None,
        preset: Optional[str] = None,  # TODO: this should actually be re-named preset_name
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        # memory
        memory: BaseMemory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_human_text(DEFAULT_PERSONA)),
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
    ) -> AgentState:
        """
        Create an agent

        Args:
            name (str): Name of the agent
            tools (List[str]): List of tools (by name) to attach to the agent
            include_base_tools (bool): Whether to include base tools (default: `True`)

        Returns:
            agent_state (AgentState): State of the the created agent.
        """
        if embedding_config or llm_config:
            raise ValueError("Cannot override embedding_config or llm_config when creating agent via REST API")

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

        # TODO: distinguish between name and objects
        # TODO: add metadata
        payload = {
            "config": {
                "name": name,
                "preset": preset,
                "persona": memory.memory["persona"].value,
                "human": memory.memory["human"].value,
                "function_names": tool_names,
                "metadata": metadata,
            }
        }
        response = requests.post(f"{self.base_url}/api/agents/create", json=payload, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Status {response.status_code} - Failed to create agent: {response.text}")
        response_obj = CreateAgentResponse(**response.json())
        return self.get_agent_response_to_state(response_obj)

    def get_agent_response_to_state(self, response: Union[GetAgentResponse, CreateAgentResponse]) -> AgentState:
        # TODO: eventually remove this conversion
        llm_config = LLMConfig(
            model=response.agent_state.llm_config.model,
            model_endpoint_type=response.agent_state.llm_config.model_endpoint_type,
            model_endpoint=response.agent_state.llm_config.model_endpoint,
            model_wrapper=response.agent_state.llm_config.model_wrapper,
            context_window=response.agent_state.llm_config.context_window,
        )
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type=response.agent_state.embedding_config.embedding_endpoint_type,
            embedding_endpoint=response.agent_state.embedding_config.embedding_endpoint,
            embedding_model=response.agent_state.embedding_config.embedding_model,
            embedding_dim=response.agent_state.embedding_config.embedding_dim,
            embedding_chunk_size=response.agent_state.embedding_config.embedding_chunk_size,
        )
        agent_state = AgentState(
            id=response.agent_state.id,
            name=response.agent_state.name,
            user_id=response.agent_state.user_id,
            llm_config=llm_config,
            embedding_config=embedding_config,
            state=response.agent_state.state,
            system=response.agent_state.system,
            tools=response.agent_state.tools,
            _metadata=response.agent_state.metadata,
            # load datetime from timestampe
            created_at=datetime.datetime.fromtimestamp(response.agent_state.created_at, tz=datetime.timezone.utc),
        )
        return agent_state

    def rename_agent(self, agent_id: uuid.UUID, new_name: str):
        response = requests.patch(f"{self.base_url}/api/agents/{str(agent_id)}/rename", json={"agent_name": new_name}, headers=self.headers)
        assert response.status_code == 200, f"Failed to rename agent: {response.text}"
        response_obj = GetAgentResponse(**response.json())
        return self.get_agent_response_to_state(response_obj)

    def delete_agent(self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None):
        """Delete the agent."""
        data = {"agent_id": str(agent_id), "agent_name": agent_name}
        response = requests.delete(f"{self.base_url}/api/agents/delete", json=data, headers=self.headers)
        assert response.status_code == 200, f"Failed to delete agent: {response.text}"

    def get_agent_config(self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None) -> AgentState:
        params = {"agent_id": agent_id, "agent_name": agent_name}
        response = requests.get(f"{self.base_url}/api/agents/config", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to get agent: {response.text}"
        response_obj = GetAgentResponse(**response.json())
        return self.get_agent_response_to_state(response_obj)

    def update_agent(self, agent_state: AgentState):
        # create json object with the modifiable fields from cli "agent_state."

        data = {
            "agent_id": agent_state.id.int,
            "agent_name": agent_state.name,
            "persona": agent_state.persona,
            "human": agent_state.human,
            "model": agent_state.llm_config.model,
            "context_window": agent_state.llm_config.context_window,
            "model_wrapper": agent_state.llm_config.model_wrapper,
            "model_endpoint": agent_state.llm_config.model_endpoint,
            "model_endpoint_type": agent_state.llm_config.model_endpoint_type,
        }
        response = requests.post(f"{self.base_url}/api/agents/update", json=data, headers=self.headers)
        assert response.status_code == 200, f"Failed to update agent: {response.text}"

    def save_agent(self, agent: Agent):
        llm_config_model = LLMConfigModel(**agent.agent_state.llm_config.__dict__)
        embedding_config_model = EmbeddingConfigModel(**agent.agent_state.embedding_config.__dict__)
        agent_state_model = AgentStateModel(
            name=agent.agent_state.name,
            user_id=str(agent.agent_state.user_id),
            persona=agent.agent_state.persona,
            human=agent.agent_state.human,
            llm_config=llm_config_model,
            embedding_config=embedding_config_model,
            preset=agent.agent_state.preset,
            id=str(agent.agent_state.id),
            state=agent.agent_state.state,
            created_at=int(agent.agent_state.created_at.timestamp()),
            functions_schema=agent.agent_state.state["functions"],
        )
        data = agent_state_model.model_dump()
        data["id"] = str(data["id"])
        data["user_id"] = str(data["user_id"])
        print("client save_agent data: ", data)
        response = requests.post(f"{self.base_url}/api/agents/save", json=data, headers=self.headers)
        assert response.status_code == 200, f"Failed to save agent: {response.text}"

    # users
    def get_user(self) -> Optional[UserModel]:
        raise NotImplementedError

    # memory
    def get_agent_memory(self, agent_id: uuid.UUID) -> GetAgentMemoryResponse:
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/memory", headers=self.headers)
        return GetAgentMemoryResponse(**response.json())

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> UpdateAgentMemoryResponse:
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/memory", json=new_memory_contents, headers=self.headers)
        return UpdateAgentMemoryResponse(**response.json())

    # agent interactions
    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        return self.send_message(agent_id, message, role="user")

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        response = requests.post(f"{self.base_url}/api/agents/{str(agent_id)}/command", json={"command": command}, headers=self.headers)
        return CommandResponse(**response.json())

    def save(self):
        raise NotImplementedError

    # archival memory

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Paginated get for the archival memory for an agent"""
        params = {"limit": limit}
        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)
        response = requests.get(f"{self.base_url}/api/agents/{str(agent_id)}/archival", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to get archival memory: {response.text}"
        return GetAgentArchivalMemoryResponse(**response.json())

    def insert_archival_memory(self, agent_id: uuid.UUID, memory: str) -> GetAgentArchivalMemoryResponse:
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/archival", json={"content": memory}, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to insert archival memory: {response.text}")
        return InsertAgentArchivalMemoryResponse(**response.json())

    def delete_archival_memory(self, agent_id: uuid.UUID, memory_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/api/agents/{agent_id}/archival?id={memory_id}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete archival memory: {response.text}"

    # messages (recall memory)

    def get_messages(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ) -> GetAgentMessagesResponse:
        params = {"before": before, "after": after, "limit": limit}
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/messages-cursor", params=params, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get messages: {response.text}")
        return GetAgentMessagesResponse(**response.json())

    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False) -> UserMessageResponse:
        data = {"message": message, "role": role, "stream": stream}
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/messages", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to send message: {response.text}")
        return UserMessageResponse(**response.json())

    # humans / personas

    def list_humans(self) -> ListHumansResponse:
        response = requests.get(f"{self.base_url}/api/humans/all", headers=self.headers)
        return ListHumansResponse(**response.json()).humans

    def get_human(self, name: str, user_id: uuid.UUID):
        response = requests.get(f"{self.base_url}/api/humans/{name}", headers=self.headers)
        if response.json() is not None:
            return HumanModel(**response.json())
        return None

    def add_human(self, name: str, text: str) -> HumanModel:
        data = {"name": name, "text": text}
        response = requests.post(f"{self.base_url}/api/humans/add", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create human: {response.text}")
        print(response.json())
        return HumanModel(**response.json())

    def update_human(self, name: str, text: str) -> HumanModel:
        data = {"name": name, "text": text}
        response = requests.post(f"{self.base_url}/api/humans/update", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create human: {response.text}")
        print(response.json())
        return HumanModel(**response.json())

    def delete_human(self, name: str, user_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/api/humans/delete/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create human: {response.text}")

    def list_personas(self) -> ListPersonasResponse:
        response = requests.get(f"{self.base_url}/api/personas/all", headers=self.headers)
        return ListPersonasResponse(**response.json()).personas

    def get_persona(self, name: str, user_id: uuid.UUID):
        response = requests.get(f"{self.base_url}/api/personas/{name}", headers=self.headers)
        if response.json() is not None:
            return PersonaModel(**response.json())
        return None

    def add_persona(self, name: str, persona: str) -> PersonaModel:
        data = {"name": name, "text": persona}
        response = requests.post(f"{self.base_url}/api/personas/add", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create persona: {response.text}")
        return PersonaModel(**response.json())

    def update_persona(self, name: str, persona: str) -> PersonaModel:
        data = {"name": name, "text": persona}
        response = requests.post(f"{self.base_url}/api/personas/update", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create persona: {response.text}")
        print(response.json())
        return PersonaModel(**response.json())

    def delete_persona(self, name: str, user_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/api/personas/delete/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create persona: {response.text}")

    def get_persona(self, name: str) -> PersonaModel:
        response = requests.get(f"{self.base_url}/api/personas/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get persona: {response.text}")
        return PersonaModel(**response.json())

    def get_human(self, name: str) -> HumanModel:
        response = requests.get(f"{self.base_url}/api/humans/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get human: {response.text}")
        return HumanModel(**response.json())

    # sources

    def list_sources(self):
        """List loaded sources"""
        response = requests.get(f"{self.base_url}/api/sources", headers=self.headers)
        response_json = response.json()
        sources = ListSourcesResponse(**response_json).sources
        for source in sources:
            source.user_id = uuid.UUID(source.user_id)
            date_format = "%Y-%m-%dT%H:%M:%S.%f"
            source.created_at = datetime.datetime.strptime(source.created_at, date_format)
            source.embedding_config = EmbeddingConfig(
                embedding_endpoint_type=source.embedding_config["embedding_endpoint_type"],
                embedding_endpoint=source.embedding_config["embedding_endpoint"],
                embedding_dim=source.embedding_config["embedding_dim"],
                embedding_model=source.embedding_config["embedding_model"],
                embedding_chunk_size=source.embedding_config["embedding_chunk_size"],
            )
        return sources

    def delete_source(self, source_name: str):
        """Delete a source and associated data (including attached to agents)"""

        get_sources = requests.get(f"{self.base_url}/api/sources", headers=self.headers)
        sources_json = get_sources.json()
        sources = ListSourcesResponse(**sources_json).sources
        source_id = uuid.UUID(int=0)
        for source in sources:
            if source.name == source_name:
                source_id = source.id
                break
        response = requests.delete(f"{self.base_url}/api/sources/{str(source_id)}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete source: {response.text}"

    def get_job_status(self, job_id: uuid.UUID):
        response = requests.get(f"{self.base_url}/api/sources/status/{str(job_id)}", headers=self.headers)
        return JobModel(**response.json())

    def load_file_into_source(self, filename: str, source_id: uuid.UUID, blocking=True):
        """Load {filename} and insert into source"""
        files = {"file": open(filename, "rb")}

        # create job
        response = requests.post(f"{self.base_url}/api/sources/{source_id}/upload", files=files, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to upload file to source: {response.text}")

        job = JobModel(**response.json())
        if blocking:
            # wait until job is completed
            while True:
                job = self.get_job_status(job.id)
                if job.status == JobStatus.completed:
                    break
                elif job.status == JobStatus.failed:
                    raise ValueError(f"Job failed: {job.metadata}")
                time.sleep(1)
        return job

    def create_source(self, name: str, description: Optional[str]) -> Source:
        """Create a new source"""
        payload = {"name": name, "description": description}
        response = requests.post(f"{self.base_url}/api/sources", json=payload, headers=self.headers)
        response_obj = SourceModel(**response.json())
        return Source(
            id=uuid.UUID(response_obj.id),
            name=response_obj.name,
            user_id=uuid.UUID(response_obj.user_id),
            created_at=response_obj.created_at,
            embedding_dim=response_obj.embedding_config["embedding_dim"],
            embedding_model=response_obj.embedding_config["embedding_model"],
        )

    def attach_source_to_agent(self, agent_id: uuid.UUID, source_id: Optional[uuid.UUID], source_name: Optional[str]):
        """Attach a source to an agent"""
        data = {"agent_id": agent_id, "source_id": source_id, "source_name": source_name}
        response = requests.post(f"{self.base_url}/api/sources/attach", json=data, headers=self.headers)
        assert response.status_code == 200, f"Failed to attach source to agent: {response.text}"

    def detach_source(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Detach a source from an agent"""
        params = {"agent_id": str(agent_id)}
        response = requests.post(f"{self.base_url}/api/sources/{source_id}/detach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to detach source from agent: {response.text}"

    # server configuration commands

    def list_models(self) -> ListModelsResponse:
        response = requests.get(f"{self.base_url}/api/models", headers=self.headers)
        return ListModelsResponse(**response.json())

    def get_config(self) -> MemGPTConfig:
        response = requests.get(f"{self.base_url}/api/config", headers=self.headers)
        json_response = response.json()
        config = ConfigResponse(**json_response).config
        memgpt_config = MemGPTConfig(**config)
        memgpt_config.default_llm_config = LLMConfig(**memgpt_config.default_llm_config)
        memgpt_config.default_embedding_config = EmbeddingConfig(**memgpt_config.default_embedding_config)
        return memgpt_config

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

        # create data
        data = {"source_code": source_code, "source_type": source_type, "tags": tags, "json_schema": json_schema, "update": update}
        try:
            CreateToolRequest(**data)  # validate data
        except Exception as e:
            raise ValueError(f"Failed to create tool: {e}, invalid input {data}")

        # make REST request
        response = requests.post(f"{self.base_url}/api/tools", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create tool: {response.text}")
        return ToolModel(**response.json())

    def list_tools(self) -> ListToolsResponse:
        response = requests.get(f"{self.base_url}/api/tools", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list tools: {response.text}")
        return ListToolsResponse(**response.json()).tools

    def delete_tool(self, name: str):
        response = requests.delete(f"{self.base_url}/api/tools/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to delete tool: {response.text}")
        return response.json()

    def get_tool(self, name: str):
        response = requests.get(f"{self.base_url}/api/tools/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return ToolModel(**response.json())

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

        # create data
        data = {"source_code": source_code, "source_type": source_type, "tags": tags, "json_schema": json_schema, "update": update}
        try:
            CreateToolRequest(**data)  # validate data
        except Exception as e:
            raise ValueError(f"Failed to create tool: {e}, invalid input {data}")

        # make REST request
        response = requests.post(f"{self.base_url}/api/tools", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create tool: {response.text}")
        return ToolModel(**response.json())

    def list_tools(self) -> ListToolsResponse:
        response = requests.get(f"{self.base_url}/api/tools", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to list tools: {response.text}")
        return ListToolsResponse(**response.json()).tools

    def delete_tool(self, name: str):
        response = requests.delete(f"{self.base_url}/api/tools/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to delete tool: {response.text}")
        return response.json()

    def get_tool(self, name: str):
        response = requests.get(f"{self.base_url}/api/tools/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return ToolModel(**response.json())


class LocalClient(AbstractClient):
    def __init__(
        self,
        auto_save: bool = False,
        user_id: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.
        :param auto_save: indicates whether to automatically save after every message.
        :param quickstart: allows running quickstart on client init.
        :param config: optional config settings to apply after quickstart
        :param debug: indicates whether to display debug messages.
        """
        self.auto_save = auto_save

        # determine user_id (pulled from local config)
        config = MemGPTConfig.load()
        if user_id:
            self.user_id = uuid.UUID(user_id)
        else:
            self.user_id = uuid.UUID(config.anon_clientid)

        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface_factory=lambda: self.interface)

        # create user if does not exist
        self.server.create_user({"id": self.user_id}, exists_ok=True)

    # agents

    def list_agents(self) -> List[AgentState]:
        self.interface.clear()
        agents_data = self.server.list_agents(user_id=self.user_id)
        return ListAgentsResponse(**agents_data).agents

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
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
        memory: BaseMemory = ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_human_text(DEFAULT_PERSONA)),
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        # metadata
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
    ) -> AgentState:
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
            tool = self.create_tool(func, name=func_name, tags=["memory", "memgpt-base"])
            tool_names.append(tool.name)

        self.interface.clear()

        # create agent
        agent_state = self.server.create_agent(
            user_id=self.user_id,
            name=name,
            memory=memory,
            llm_config=llm_config,
            embedding_config=embedding_config,
            tools=tool_names,
            metadata=metadata,
        )
        return agent_state

    def rename_agent(self, agent_id: uuid.UUID, new_name: str):
        # TODO: check valid name
        agent_state = self.server.rename_agent(user_id=self.user_id, agent_id=agent_id, new_agent_name=new_name)
        return agent_state

    def delete_agent(self, agent_id: uuid.UUID):
        self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)

    def get_agent(self, agent_id: uuid.UUID) -> AgentState:
        self.interface.clear()
        return self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id, agent_name=agent_name)

    def update_agent(self, agent_state: AgentState):
        self.server.update_agent(agent_state)

    def save_agent(self, agent: Agent):
        self.server.save_agent(agent)

    def delete_agent(self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None):
        self.server.delete_agent(user_id=self.user_id, agent_id=agent_id, agent_name=agent_name)

    # users
    def get_user(self) -> Optional[UserModel]:
        return self.server.get_user(self.user_id)

    # memory
    def get_agent_memory(self, agent_id: str) -> Dict:
        memory = self.server.get_agent_memory(user_id=self.user_id, agent_id=agent_id)
        return GetAgentMemoryResponse(**memory)

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
        self.interface.clear()
        return self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)

    # agent interactions

    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False) -> UserMessageResponse:
        if stream:
            # TODO: implement streaming with stream=True/False
            raise NotImplementedError
        self.interface.clear()
        if role == "system":
            usage = self.server.system_message(user_id=self.user_id, agent_id=agent_id, message=message)
        elif role == "user":
            usage = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        else:
            raise ValueError(f"Role {role} not supported")
        if self.auto_save:
            self.save()
        else:
            return UserMessageResponse(messages=self.interface.to_list(), usage=usage)

    def user_message(self, agent_id: str, message: str) -> UserMessageResponse:
        self.interface.clear()
        usage = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        else:
            return UserMessageResponse(messages=self.interface.to_list(), usage=usage)

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        self.interface.clear()
        return self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

    def save(self):
        self.server.save_agents()

    # archival memory
    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        _, archival_json_records = self.server.get_agent_archival_cursor(
            user_id=self.user_id,
            agent_id=agent_id,
            after=after,
            before=before,
            limit=limit,
        )
        return archival_json_records

    # messages (recall memory)
    # (No corresponding function in LocalClient)

    # humans / personas

    def create_human(self, name: str, human: str):
        return self.server.add_human(HumanModel(name=name, text=human, user_id=self.user_id))

    def create_persona(self, name: str, persona: str):
        return self.server.add_persona(PersonaModel(name=name, text=persona, user_id=self.user_id))

    def list_humans(self):
        return self.server.list_humans(user_id=self.user_id if self.user_id else self.user_id)

    def get_human(self, name: str):
        return self.server.get_human(name=name, user_id=self.user_id)

    def update_human(self, name: str, text: str):
        return self.server.update_human(name, text, self.user_id)

    def delete_human(self, name: str):
        return self.server.delete_human(name, self.user_id)

    def list_personas(self):
        return self.server.list_personas(user_id=self.user_id)

    def get_persona(self, name: str):
        return self.server.get_persona(name=name, user_id=self.user_id)

    def update_persona(self, persona: PersonaModel):
        return self.server.update_persona(persona=persona)

    def delete_persona(self, name: str):
        return self.server.delete_persona(name, self.user_id)

    # tools
    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        update: Optional[bool] = True,  # TODO: actually use this
        tags: Optional[List[str]] = None,
    ):
        """
        Create a tool.

        Args:
            func (callable): The function to create a tool for.
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            update (bool, optional): Update the tool if it already exists. Defaults to True.

        Returns:
            tool (ToolModel): The created tool.
        """

        # TODO: check if tool already exists
        # TODO: how to load modules?
        # parse source code/schema
        source_code = parse_source_code(func)
        json_schema = generate_schema(func, name)
        source_type = "python"
        tool_name = json_schema["name"]

        assert name is None or name == tool_name, f"Tool name {name} does not match schema name {tool_name}"

        return self.server.create_tool(
            user_id=self.user_id,
            source_code=source_code,
            source_type=source_type,
            tags=tags,
            json_schema=json_schema,
            exists_ok=update,
        )

    def list_tools(self):
        """List available tools.

        Returns:
            tools (List[ToolModel]): A list of available tools.

        """
        return self.server.ms.list_tools(user_id=self.user_id)

    def get_tool(self, name: str):
        return self.server.ms.get_tool(name, user_id=self.user_id)

    def delete_tool(self, name: str):
        return self.server.ms.delete_tool(name, user_id=self.user_id)

    # data sources

    def add_persona(self, name: str, text: str):
        new_persona = PersonaModel(text=text, name=name, user_id=self.user_id)
        self.server.add_persona(persona=new_persona)

    def update_persona(self, name: str, text: str):
        self.server.update_persona(name, text, self.user_id)

    def delete_persona(self, name: str, user_id: uuid.UUID):
        self.server.delete_persona(name, user_id)

    # sources
    def list_sources(self):
        return self.server.list_all_sources(user_id=self.user_id)

    def create_source(self, name: str, description: Optional[str]):
        return self.server.create_source(user_id=self.user_id, name=name, description=description)

    def attach_source_to_agent(self, agent_id: uuid.UUID, source_id: Optional[uuid.UUID], source_name: Optional[str]):
        self.server.attach_source_to_agent(user_id=self.user_id, source_id=source_id, source_name=source_name, agent_id=agent_id)

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        self.interface.clear()
        # TODO need to add support for non-postgres here
        # chroma will throw:
        #     raise ValueError("Cannot run get_all_cursor with chroma")
        _, archival_json_records = self.server.get_agent_archival_cursor(
            user_id=self.user_id,
            agent_id=agent_id,
            after=after,
            before=before,
            limit=limit,
        )
        archival_memory_objects = [ArchivalMemoryObject(id=passage["id"], contents=passage["text"]) for passage in archival_json_records]
        return GetAgentArchivalMemoryResponse(archival_memory=archival_memory_objects)

    def insert_archival_memory(self, agent_id: uuid.UUID, memory: str) -> GetAgentArchivalMemoryResponse:
        memory_ids = self.server.insert_archival_memory(user_id=self.user_id, agent_id=agent_id, memory_contents=memory)
        return InsertAgentArchivalMemoryResponse(ids=memory_ids)

    def delete_archival_memory(self, agent_id: uuid.UUID, memory_id: uuid.UUID):
        self.server.delete_archival_memory(user_id=self.user_id, agent_id=agent_id, memory_id=memory_id)

    def get_messages(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ) -> GetAgentMessagesResponse:
        self.interface.clear()
        [_, messages] = self.server.get_agent_recall_cursor(
            user_id=self.user_id, agent_id=agent_id, before=before, limit=limit, reverse=True
        )
        return GetAgentMessagesResponse(messages=messages)

    def list_models(self) -> ListModelsResponse:

        llm_config = LLMConfigModel(
            model=self.server.server_llm_config.model,
            model_endpoint=self.server.server_llm_config.model_endpoint,
            model_endpoint_type=self.server.server_llm_config.model_endpoint_type,
            model_wrapper=self.server.server_llm_config.model_wrapper,
            context_window=self.server.server_llm_config.context_window,
        )

        return ListModelsResponse(models=[llm_config])

    def delete_source(self, source_id: Optional[uuid.UUID] = None, source_name: Optional[str] = None):
        self.server.delete_source(user_id=self.user.id, source_id=source_id, source_name=source_name)

    def get_source(self, source_id: Optional[uuid.UUID] = None, source_name: Optional[str] = None):
        return self.server.ms.get_source(user_id=self.user_id, source_id=source_id, source_name=source_name)

    # server configuration commands

    # def list_models(self) -> ListModelsResponse:
    #     response = requests.get(f"{self.base_url}/api/models", headers=self.headers)
    #     return ListModelsResponse(**response.json())

    def get_config(self) -> MemGPTConfig:
        config = self.server.get_server_config()["config"]
        memgpt_config = MemGPTConfig(**config)
        memgpt_config.default_llm_config = LLMConfig(**memgpt_config.default_llm_config)
        memgpt_config.default_embedding_config = EmbeddingConfig(**memgpt_config.default_embedding_config)
        return memgpt_config
