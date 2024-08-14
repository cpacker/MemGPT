import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import httpx

from memgpt.config import MemGPTConfig
from memgpt.log import get_logger
from memgpt.constants import BASE_TOOLS
from memgpt.settings import settings
from memgpt.data_sources.connectors import DataConnector
from memgpt.functions.functions import parse_source_code
from memgpt.functions.schema_generator import generate_schema
from memgpt.memory import get_memory_functions

# This is a hack for now, should be using new schemas
from memgpt.server.schemas.humans import ListHumansResponse
from memgpt.server.schemas.personas import ListPersonasResponse
from memgpt.server.schemas.config import ConfigResponse

# new schemas
from memgpt.schemas.block import Human, Persona
from memgpt.schemas.agent import AgentState, CreateAgent, UpdateAgentState
from memgpt.schemas.block import (
    Block,
    CreateBlock,
    CreateHuman,
    CreatePersona,
    Human,
    Persona,
    UpdateHuman,
    UpdatePersona,
)
from memgpt.schemas.embedding_config import EmbeddingConfig

# new schemas
from memgpt.schemas.enums import JobStatus
from memgpt.schemas.job import Job
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.memgpt_request import MemGPTRequest
from memgpt.schemas.memgpt_response import MemGPTResponse
from memgpt.schemas.memory import (
    ArchivalMemorySummary,
    ChatMemory,
    Memory,
    RecallMemorySummary,
)
from memgpt.schemas.job import Job
from memgpt.schemas.enums import JobStatus
from memgpt.schemas.message import Message
from memgpt.schemas.passage import Passage
from memgpt.schemas.source import Source, SourceCreate, SourceUpdate
from memgpt.schemas.tool import Tool, ToolCreate, ToolUpdate
from memgpt.schemas.user import UserCreate
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.utils import get_human_text

if TYPE_CHECKING:
    from httpx import ASGITransport, WSGITransport

logger = get_logger(__name__)

def create_client(base_url: Optional[str] = None,
                  token: Optional[str] = None,
                  config: Optional[MemGPTConfig] = None,
                  app: Optional[str] = None,
                  debug: Optional[bool] = False) -> Union["RESTClient", "LocalClient"]:
    """factory method to create either a local or rest api enabled client.
    _TODO: link to docs on the difference between the two._

    base_url: str if provided, the url to the rest api server
    token: str if provided, the token to authenticate to the rest api server
    config: MemGPTConfig if provided, the configuration settings to use for the local client
    app: str if provided an ASGI compliant application to use instead of an actual http call. used for testing hook.
    """
    if base_url:
        return RESTClient(base_url=base_url, token=token, debug=debug, app=app)
    return LocalClient(config=config, debug=debug)


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

    def agent_exists(self, agent_id: Optional[str] = None) -> bool:
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
        memory: Optional[Memory] = None,
    ) -> AgentState:
        """Create a new agent with the specified configuration."""
        raise NotImplementedError

    def rename_agent(self, agent_id: str, new_name: str):
        """Rename the agent."""
        raise NotImplementedError

    def delete_agent(self, agent_id: str):
        """Delete the agent."""
        raise NotImplementedError

    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        raise NotImplementedError

    # memory

    def get_in_context_memory(self, agent_id: str) -> Dict:
        raise NotImplementedError

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        raise NotImplementedError

    # agent interactions

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    # archival memory

    def get_archival_memory(self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000):
        """Paginated get for the archival memory for an agent"""
        raise NotImplementedError

    def insert_archival_memory(self, agent_id: str, memory: str):
        """Insert archival memory into the agent."""
        raise NotImplementedError

    def delete_archival_memory(self, agent_id: str, memory_id: str):
        """Delete archival memory from the agent."""
        raise NotImplementedError

    # messages (recall memory)

    def get_messages(self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000):
        """Get messages for the agent."""
        raise NotImplementedError

    def send_message(self, agent_id: str, message: str, role: str, stream: Optional[bool] = False):
        """Send a message to the agent."""
        raise NotImplementedError

    # humans / personas

    def list_humans(self):
        """List all humans."""
        raise NotImplementedError

    def create_human(self, name: str, text: str):
        """Create a human."""
        raise NotImplementedError

    def list_personas(self):
        """List all personas."""
        raise NotImplementedError

    def create_persona(self, name: str, text: str):
        """Create a persona."""
        raise NotImplementedError

    # tools

    def list_tools(self):
        """List all tools."""
        raise NotImplementedError

    # data sources

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

    # server configuration commands

    def list_models(self):
        """List all models."""
        raise NotImplementedError

    def get_config(self):
        """Get server config"""
        raise NotImplementedError


class RESTClient(AbstractClient):

    httpx_client: "httpx.Client"

    def __init__(
        self,
        base_url: str,
        token: str,
        debug: bool = False,
        app: Optional[Union["WSGITransport","ASGITransport"]] = None,
    ):
        super().__init__(debug=debug)

        httpx_client_args = {
            "headers": {"accept": "application/json", "authorization": f"Bearer {token}"},
            "base_url": base_url,
        }
        if app:
            logger.warning("Using supplied WSGI or ASGI app for RESTClient")
            httpx_client_args["app"] = app

        self.httpx_client = httpx.AsyncClient(**httpx_client_args)

    async def list_agents(self) -> List[AgentState]:
        response = await self.httpx_client.get("/api/agents")
        return response.json()

    async def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        response = await self.httpx_client.get("/agents/")
        if response.status_code != 200:
            raise ValueError(f"Failed to list agents: {response.text}")
        for agent in response.json()["agents"]:
            if agent_id and agent["id"] == agent_id:
                return True
            if agent_name and agent["name"] == agent_name:
                return True
        return False


    def get_tool(self, tool_name: str):
        response = self.httpx_client.get(f"/api/tools/{tool_name}")
        if response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return Tool(**response.json())

    async def create_agent(
        self,
        name: Optional[str] = None,
        # model configs
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        # memory
        memory: Memory = ChatMemory(human=get_human_text(settings.human), persona=get_human_text(settings.persona)),
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        metadata: Optional[Dict] = {"human:": settings.human, "persona": settings.persona},
        description: Optional[str] = None,
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
        tool_names = []
        if tools:
            tool_names += tools
        if include_base_tools:
            tool_names += BASE_TOOLS

        # TODO: why is this here?
        # add memory tools
        memory_functions = get_memory_functions(memory)
        for func_name, func in memory_functions.items():
            tool = await self.create_tool(func, name=func_name, tags=["memory", "memgpt-base"], update=True)
            tool_names.append(tool.name)

        request = CreateAgent(
            name=name,
            description=description,
            metadata_=metadata,
            memory=memory,
            tools=tool_names,
            llm_config=llm_config,
            embedding_config=embedding_config,
        )

        response = await self.httpx_client.post("/agents", json=request.model_dump(exclude_none=True))
        if response.status_code != 200:
            raise ValueError(f"Status {response.status_code} - Failed to create agent: {response.text}")

        return AgentState(**response.json())


    def rename_agent(self, agent_id: str, new_name: str):
        response = self.httpx_client.patch(f"/api/agents/{agent_id}/rename", json={"agent_name": new_name})
        assert response.status_code == 200, f"Failed to rename agent: {response.text}"

        return AgentState(**response.json())

    async def update_agent(
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
        response = await self.httpx_client.post(f"/agents/{agent_id}", json=request.model_dump(exclude_none=True))
        if response.status_code != 200:
            raise ValueError(f"Failed to update agent: {response.text}")
        return AgentState(**response.json())

    def rename_agent(self, agent_id: str, new_name: str):
        return self.update_agent(agent_id, name=new_name)

    def delete_agent(self, agent_id: str):
        """Delete the agent."""
        response = self.httpx_client.delete("/api/agents/{str(agent_id)}")
        assert response.status_code == 200, f"Failed to delete agent: {response.text}"

    async def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        response = await self.httpx_client.get(f"/agents/{agent_id}/config")
        # TODO: this should be a 404 without details, don't share failed response with a bad actor
        assert response.status_code == 200, f"Failed to get agent: {response.text}"

        return AgentState(**response.json())

    # memory
    async def get_agent_memory(self, agent_id: uuid.UUID) -> Memory:
        response = await self.httpx_client.get(f"/agents/{agent_id}/memory")
        return Memory(**response.json())

    async def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Memory:
        response = await self.httpx_client.post(f"/agents/{agent_id}/memory", json=new_memory_contents)
        return Memory(**response.json())

    # memory
    async def get_in_context_memory(self, agent_id: uuid.UUID) -> Memory:
        response = await self.httpx_client.get(f"/agents/{agent_id}/memory")
        if response.status_code != 200:
            raise ValueError(f"Failed to get in-context memory: {response.text}")
        return Memory(**response.json())

    async def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        memory_update_dict = {section: value}
        response = await self.httpx_client.post(f"/agents/{agent_id}/memory", json=memory_update_dict)
        if response.status_code != 200:
            raise ValueError(f"Failed to update in-context memory: {response.text}")
        return Memory(**response.json())

    async def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        response = await self.httpx_client.get(f"/agents/{agent_id}/memory/archival")
        if response.status_code != 200:
            raise ValueError(f"Failed to get archival memory summary: {response.text}")
        return ArchivalMemorySummary(**response.json())

    async def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        response = await self.httpx_client.get(f"/agents/{agent_id}/memory/recall")
        if response.status_code != 200:
            raise ValueError(f"Failed to get recall memory summary: {response.text}")
        return RecallMemorySummary(**response.json())

    async def get_in_context_messages(self, agent_id: str) -> List[Message]:
        response = await self.httpx_client.get(f"/agents/{agent_id}/memory/messages")
        if response.status_code != 200:
            raise ValueError(f"Failed to get in-context messages: {response.text}")
        return [Message(**message) for message in response.json()]

    # agent interactions

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        return self.send_message(agent_id, message, role="user")

    def run_command(self, agent_id: str, command: str) -> Union[Message, str, None]:
        response = self.httpx_client.post(f"/api/agents/{agent_id}/command", json={"command": command})
        return Message(**response.json())

    def save(self):
        raise NotImplementedError

    # archival memory

    def get_archival_memory(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Passage]:
        """Paginated get for the archival memory for an agent"""
        params = {"limit": limit}
        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)
        response = self.httpx_client.get("/api/agents/{str(agent_id)}/archival", params=params)
        assert response.status_code == 200, f"Failed to get archival memory: {response.text}"
        return [Passage(**passage) for passage in response.json()]

    async def insert_archival_memory(self, agent_id: uuid.UUID, memory: str) -> List[Passage]:
        response = await self.httpx_client.post(f"/agents/{agent_id}/archival/{memory}")
        if response.status_code != 200:
            raise ValueError(f"Failed to insert archival memory: {response.text}")
        return [Passage(**passage) for passage in response.json()]

    async def delete_archival_memory(self, agent_id: uuid.UUID, memory_id: uuid.UUID):
        response = await self.httpx_client.delete("/agents/{agent_id}/archival?id={memory_id}")
        assert response.status_code == 200, f"Failed to delete archival memory: {response.text}"

    # messages (recall memory)

    def get_messages(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> MemGPTResponse:
        params = {"before": before, "after": after, "limit": limit}
        response = self.httpx_client.get("/api/agents/{agent_id}/messages-cursor", params=params)
        if response.status_code != 200:
            raise ValueError(f"Failed to get messages: {response.text}")
        return [Message(**message) for message in response.json()]

    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False) -> MemGPTResponse:
        data = {"message": message, "role": role, "stream": stream}
        response = self.httpx_client.post("/api/agents/{agent_id}/messages", json=data)
        if response.status_code != 200:
            raise ValueError(f"Failed to send message: {response.text}")
        return MemGPTResponse(**response.json())

    # humans / personas

    async def list_humans(self) -> ListHumansResponse:
        response = await self.httpx_client.get("/api/humans")
        return ListHumansResponse(**response.json())

    def create_human(self, name: str, human: str) -> Human:
        data = {"name": name, "text": human}
        response = self.httpx_client.post("/api/humans", json=data)

    async def list_blocks(self, label: Optional[str] = None, templates_only: Optional[bool] = True) -> List[Block]:
        params = {"label": label, "templates_only": templates_only}
        response = await self.httpx_client.get(f"/blocks", params=params)
        if response.status_code != 200:
            raise ValueError(f"Failed to list blocks: {response.text}")
        if label == "human":
            return [Human(**human) for human in response.json()]
        elif label == "persona":
            return [Persona(**persona) for persona in response.json()]
        else:
            return [Block(**block) for block in response.json()]

    async def list_personas(self) -> ListPersonasResponse:
        response = await self.httpx_client.get("/personas")
        return ListPersonasResponse(**response.json())

    async def create_persona(self, name: str, persona: str) -> Persona:
        data = {"name": name, "text": persona}
        response = await self.httpx_client.post("/personas", json=data)

    async def create_block(self, label: str, name: str, text: str) -> Block:  #
        request = CreateBlock(label=label, name=name, value=text)
        response = await self.httpx_client.post(f"/blocks", json=request.model_dump())
        if response.status_code != 200:
            raise ValueError(f"Failed to create block: {response.text}")
        if request.label == "human":
            return Human(**response.json())
        elif request.label == "persona":
            return Persona(**response.json())
        else:
            return Block(**response.json())

    async def get_persona(self, name: str) -> Persona:
        response = await self.httpx_client.get("/personas/{name}")
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get persona: {response.text}")
        return Persona(**response.json())

    async def get_human(self, name: str) -> Human:
        response = await self.httpx_client.get("/api/humans/{name}")
        if response.status_code == 404:
            return None

    async def get_block(self, block_id: str) -> Block:
        response = await self.httpx_client.get(f"/blocks/{block_id}")
        if response.status_code != 200:
            raise ValueError(f"Failed to get block: {response.text}")
        return Block(**response.json())

    async def get_block_id(self, name: str, label: str) -> str:
        params = {"name": name, "label": label}
        response = await self.httpx_client.get(f"/blocks", params=params)
        if response.status_code != 200:
            raise ValueError(f"Failed to get block ID: {response.text}")
        blocks = [Block(**block) for block in response.json()]
        if not blocks:
            return None
        if len(blocks) > 1:
            raise ValueError(f"Multiple blocks found with name {name} and label {label}")
        return blocks[0].id

    async def delete_block(self, id: str) -> Block:
        response = await self.httpx_client.delete(f"/blocks/{id}")
        assert response.status_code == 200, f"Failed to delete block: {response.text}"
        if response.status_code != 200:
            raise ValueError(f"Failed to delete block: {response.text}")
        return Block(**response.json())

    def list_humans(self):
        blocks = self.list_blocks(label="human")
        return [Human(**block.model_dump()) for block in blocks]

    def create_human(self, name: str, text: str) -> Human:
        return self.create_block(label="human", name=name, text=text)

    def update_human(self, human_id: str, name: Optional[str] = None, text: Optional[str] = None) -> Human:
        request = UpdateHuman(id=human_id, name=name, value=text)
        response = requests.post(f"{self.base_url}/api/blocks/{human_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update human: {response.text}")
        return Human(**response.json())

    def list_personas(self):
        blocks = self.list_blocks(label="persona")
        return [Persona(**block.model_dump()) for block in blocks]

    def create_persona(self, name: str, text: str) -> Persona:
        return self.create_block(label="persona", name=name, text=text)

    def update_persona(self, persona_id: str, name: Optional[str] = None, text: Optional[str] = None) -> Persona:
        request = UpdatePersona(id=persona_id, name=name, value=text)
        response = requests.post(f"{self.base_url}/api/blocks/{persona_id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update persona: {response.text}")
        return Persona(**response.json())

    def get_persona(self, persona_id: str) -> Persona:
        return self.get_block(persona_id)

    def get_persona_id(self, name: str) -> str:
        return self.get_block_id(name, "persona")

    def delete_persona(self, persona_id: str) -> Persona:
        return self.delete_block(persona_id)

    def get_human(self, human_id: str) -> Human:
        return self.get_block(human_id)

    def get_human_id(self, name: str) -> str:
        return self.get_block_id(name, "human")

    def delete_human(self, human_id: str) -> Human:
        return self.delete_block(human_id)

    # sources

    async def list_sources(self) -> List[Source]:
        """List loaded sources"""
        response = await self.httpx_client.get("/api/sources")
        response_json = response.json()
        return [i for i in response_json["sources"]]

    def delete_source(self, source_id: str):
        """Delete a source and associated data (including attached to agents)"""
        response = self.httpx_client.delete(f"/api/sources/{source_id}")
        assert response.status_code == 200, f"Failed to delete source: {response.text}"

    def get_job_status(self, job_id: uuid.UUID):
        response = self.httpx_client.get(f"/api/sources/status/{job_id}")
        return Job(**response.json())

    def load_file_into_source(self, filename: str, source_id: str, blocking=True):
        """Load {filename} and insert into source"""
        files = {"file": open(filename, "rb")}

        # create job
        response = self.httpx_client.post(f"/api/sources/{source_id}/upload", files=files)
        if response.status_code != 200:
            raise ValueError(f"Failed to upload file to source: {response.text}")

        job = Job(**response.json())
        job = Job(**response.json())
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

    def create_source(self, name: str) -> Source:
        """Create a new source"""
        payload = {"name": name}
        response = self.httpx_client.post("/api/sources", json=payload)
        return Source(**response.json())

    def attach_source_to_agent(self, source_id: str, agent_id: str):
        """Attach a source to an agent"""
        params = {"agent_id": agent_id}
        response = self.httpx_client.post("/api/sources/{source_id}/attach", params=params)
        assert response.status_code == 200, f"Failed to attach source to agent: {response.text}"

    def detach_source(self, source_id: str, agent_id: str):
        """Detach a source from an agent"""
        params = {"agent_id": str(agent_id)}
        response = self.httpx_client.post("/api/sources/{source_id}/detach", params=params)
        assert response.status_code == 200, f"Failed to detach source from agent: {response.text}"

    # server configuration commands
    async def list_models(self) -> List[LLMConfig]:
        response = await self.httpx_client.get("/api/models")
        return response.json()

    async def get_config(self) -> ConfigResponse:
        response = await self.httpx_client.get("/api/config")
        return ConfigResponse(**response.json())

    # tools
    async def get_tool_id(self, tool_name: str):
        response = await self.httpx_client.get(f"/tools/name/{tool_name}")
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return response.json()

    async def create_tool(
        self,
        func,
        name: Optional[str] = None,
        update: Optional[bool] = True,  # TODO: actually use this
        tags: Optional[List[str]] = None,
    ) -> Tool:
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

        # make REST request
        request = ToolCreate(source_type=source_type, source_code=source_code, name=tool_name, json_schema=json_schema, tags=tags)
        response = await self.httpx_client.post("/tools/", json=request.model_dump(exclude_none=True))
        if response.status_code != 200:
            raise ValueError(f"Failed to create tool: {response.text}")
        return Tool(**response.json())

    async def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        func: Optional[callable] = None,
        tags: Optional[List[str]] = None,
    ) -> Tool:
        """
        Update existing tool

        Args:
            id (str): Unique ID for tool

        Returns:
            tool (Tool): Updated tool object

        """
        if func:
            source_code = parse_source_code(func)
            json_schema = generate_schema(func, name)
        else:
            source_code = None
            json_schema = None

        source_type = "python"
        tool_name = json_schema["name"] if name else name

        request = ToolUpdate(id=id, source_type=source_type, source_code=source_code, tags=tags, json_schema=json_schema, name=tool_name)
        response = await self.httpx_client.post(f"{self.base_url}/api/tools/{id}", json=request.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to update tool: {response.text}")
        return Tool(**response.json())

    async def list_tools(self) -> List[Tool]:
        response = await self.httpx_client.get("/tools")
        if response.status_code != 200:
            raise ValueError(f"Failed to list tools: {response.text}")
        return [Tool(**tool) for tool in response.json()]
        return [Tool(**tool) for tool in response.json()]

    async def delete_tool(self, name: str):
        response = await self.httpx_client.delete(f"/tools/{name}")
        if response.status_code != 200:
            raise ValueError(f"Failed to delete tool: {response.text}")

    async def get_tool(self, name: str):
        response = await self.httpx_client.get(f"/tools/{name}")
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return Tool(**response.json())
        return Tool(**response.json())


class LocalClient(AbstractClient):
    def __init__(
        self,
        auto_save: bool = False,
        user_id: Optional[str] = None,
        debug: bool = False,
        config: "MemGPTConfig" = None,
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
        config = config or MemGPTConfig.load()
        if user_id:
            self.user_id = user_id
        else:
            # TODO: find a neater way to do this
            self.user_id = config.anon_clientid

        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface_factory=lambda: self.interface)

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
        memory: Memory = ChatMemory(human=get_human_text(settings.human), persona=get_human_text(settings.persona)),
        # system
        system: Optional[str] = None,
        # tools
        tools: Optional[List[str]] = None,
        include_base_tools: Optional[bool] = True,
        # metadata
        metadata: Optional[Dict] = {"human:": settings.human, "persona": settings.persona},
        description: Optional[str] = None,
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
        return self.update_agent(agent_id, name=new_name)

    def delete_agent(self, agent_id: str):
        self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)

    def get_agent(self, agent_id: str) -> AgentState:
        # TODO: include agent_name
        self.interface.clear()
        return self.server.get_agent_state(user_id=self.user_id, agent_id=agent_id)

    def get_agent_id(self, agent_name: str) -> AgentState:
        self.interface.clear()
        assert agent_name, f"Agent name must be provided"
        return self.server.get_agent_id(name=agent_name, user_id=self.user_id)

    # memory
    def get_in_context_memory(self, agent_id: str) -> Memory:
        memory = self.server.get_agent_memory(agent_id=agent_id)
        return memory

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        # TODO: implement this (not sure what it should look like)
        memory = self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents={section: value})
        return memory

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        return self.server.get_archival_memory_summary(agent_id=agent_id)

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        return self.server.get_recall_memory_summary(agent_id=agent_id)

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        return self.server.get_in_context_messages(agent_id=agent_id)

    # agent interactions

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        stream: Optional[bool] = False,
    ) -> MemGPTResponse:
        if not agent_id:
            assert agent_name, f"Either agent_id or agent_name must be provided"
            raise NotImplementedError
            # agent_state = self.get_agent(agent_name=agent_name)
            # agent_id = agent_state.id
        agent_state = self.get_agent(agent_id=agent_id)

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

        # auto-save
        if self.auto_save:
            self.save()

        # TODO: need to make sure date/timestamp is propely passed
        # TODO: update self.interface.to_list() to return actual Message objects
        #       here, the message objects will have faulty created_by timestamps
        messages = self.interface.to_list()
        for m in messages:
            assert isinstance(m, Message), f"Expected Message object, got {type(m)}"
        return MemGPTResponse(messages=messages, usage=usage)

    def user_message(self, agent_id: str, message: str) -> MemGPTResponse:
        self.interface.clear()
        return self.send_message(role="user", agent_id=agent_id, message=message)

    def run_command(self, agent_id: str, command: str) -> MemGPTResponse:
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
        return self.server.create_block(CreateHuman(name=name, value=text, user_id=self.user_id), user_id=self.user_id)

    def create_persona(self, name: str, text: str):
        return self.server.create_block(CreatePersona(name=name, value=text, user_id=self.user_id), user_id=self.user_id)

    def list_humans(self):
        return self.server.get_blocks(label="human", user_id=self.user_id, template=True)

    def list_personas(self) -> List[Persona]:
        return self.server.get_blocks(label="persona", user_id=self.user_id, template=True)

    def update_human(self, human_id: str, text: str):
        return self.server.update_block(UpdateHuman(id=human_id, value=text, user_id=self.user_id, template=True))

    def update_persona(self, persona_id: str, text: str):
        return self.server.update_block(UpdatePersona(id=persona_id, value=text, user_id=self.user_id, template=True))

    def get_persona(self, id: str) -> Persona:
        assert id, f"Persona ID must be provided"
        return Persona(**self.server.get_block(id).model_dump())

    def get_human(self, id: str) -> Human:
        assert id, f"Human ID must be provided"
        return Human(**self.server.get_block(id).model_dump())

    def get_persona_id(self, name: str) -> str:
        persona = self.server.get_blocks(name=name, label="persona", user_id=self.user_id, template=True)
        if not persona:
            return None
        return persona[0].id

    def get_human_id(self, name: str) -> str:
        human = self.server.get_blocks(name=name, label="human", user_id=self.user_id, template=True)
        if not human:
            return None
        return human[0].id

    def delete_persona(self, id: str):
        self.server.delete_block(id)

    def delete_human(self, id: str):
        self.server.delete_block(id)

    # tools
    def add_tool(self, tool: Tool, update: Optional[bool] = True) -> None:
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

        # check if tool exists
        existing_tool_id = self.get_tool_id(tool_name)
        if existing_tool_id:
            if update:
                return self.update_tool(existing_tool_id, name=name, func=func, tags=tags)
            else:
                raise ValueError(f"Tool with name {tool_name} already exists")

        # call server function
        return self.server.create_tool(
            ToolCreate(source_type=source_type, source_code=source_code, name=tool_name, json_schema=json_schema, tags=tags),
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
        Update existing tool

        Args:
            id (str): Unique ID for tool

        Returns:
            tool (Tool): Updated tool object

        """
        if func:
            source_code = parse_source_code(func)
            json_schema = generate_schema(func, name)
        else:
            source_code = None
            json_schema = None

        source_type = "python"
        tool_name = json_schema["name"] if name else name

        return self.server.update_tool(
            ToolUpdate(id=id, source_type=source_type, source_code=source_code, tags=tags, json_schema=json_schema, name=tool_name)
        )

    def list_tools(self):
        """List available tools.

        Returns:
            tools (List[ToolModel]): A list of available tools.

        """
        tools = self.server.list_tools(user_id=self.user_id)
        return tools

    def get_tool(self, id: str) -> Tool:
        return self.server.get_tool(id)

    def delete_tool(self, id: str):
        return self.server.delete_tool(id)

    def get_tool_id(self, name: str) -> Optional[str]:
        return self.server.get_tool_id(name, self.user_id)

    # data sources

    def load_data(self, connector: DataConnector, source_name: str):
        self.server.load_data(user_id=self.user_id, connector=connector, source_name=source_name)

    def load_file_into_source(self, filename: str, source_id: str, blocking=True):
        """Load {filename} and insert into source"""
        job = self.server.create_job(user_id=self.user_id)

        # TODO: implement blocking vs. non-blocking
        self.server.load_file_to_source(source_id=source_id, file_path=filename, job_id=job.id)
        return job

    def create_source(self, name: str) -> Source:
        request = SourceCreate(name=name)
        return self.server.create_source(request=request, user_id=self.user_id)

    def delete_source(self, source_id: str):
        # TODO: delete source data
        self.server.delete_source(source_id=source_id, user_id=self.user_id)

    def get_source(self, source_id: str) -> Source:
        return self.server.get_source(source_id=source_id, user_id=self.user_id)

    def get_source_id(self, source_name: str) -> str:
        return self.server.get_source_id(source_name=source_name, user_id=self.user_id)

    def attach_source_to_agent(self, agent_id: str, source_id: Optional[str] = None, source_name: Optional[str] = None):
        self.server.attach_source_to_agent(source_id=source_id, source_name=source_name, agent_id=agent_id, user_id=self.user_id)

    def detach_source_from_agent(self, agent_id: str, source_id: Optional[str] = None, source_name: Optional[str] = None):
        self.server.detach_source_from_agent(source_id=source_id, source_name=source_name, agent_id=agent_id, user_id=self.user_id)

    def list_sources(self) -> List[Source]:
        return self.server.list_all_sources(user_id=self.user_id)

    def list_attached_sources(self, agent_id: str) -> List[Source]:
        return self.server.list_attached_sources(agent_id=agent_id)

    def update_source(self, source_id: str, name: Optional[str] = None) -> Source:
        # TODO should the arg here just be "source_update: Source"?
        request = SourceUpdate(id=source_id, name=name)
        return self.server.update_source(request=request, user_id=self.user_id)

    # archival memory

    def insert_archival_memory(self, agent_id: str, memory: str) -> List[Passage]:
        return self.server.insert_archival_memory(user_id=self.user_id, agent_id=agent_id, memory_contents=memory)

    def delete_archival_memory(self, agent_id: str, memory_id: str):
        self.server.delete_archival_memory(user_id=self.user_id, agent_id=agent_id, memory_id=memory_id)

    def get_archival_memory(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Passage]:
        return self.server.get_agent_archival_cursor(user_id=self.user_id, agent_id=agent_id, before=before, after=after, limit=limit)

    # recall memory

    def get_messages(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Message]:
        self.interface.clear()
        return self.server.get_agent_recall_cursor(
            user_id=self.user_id, agent_id=agent_id, before=before, after=after, limit=limit, reverse=True
        )

    def list_models(self) -> List[LLMConfig]:
        return [self.server.server_llm_config]

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        return [self.server.server_embedding_config]
