# inspecting tools
import json
import os
import traceback
import warnings
from abc import abstractmethod
from asyncio import Lock
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

from composio.client import Composio
from composio.client.collections import ActionModel, AppModel
from fastapi import HTTPException

import letta.constants as constants
import letta.server.utils as server_utils
import letta.system as system
from letta.agent import Agent, save_agent
from letta.agent_store.db import attach_base
from letta.agent_store.storage import StorageConnector, TableType
from letta.chat_only_agent import ChatOnlyAgent
from letta.credentials import LettaCredentials
from letta.data_sources.connectors import DataConnector, load_data
from letta.errors import LettaAgentNotFoundError, LettaUserNotFoundError

# TODO use custom interface
from letta.interface import AgentInterface  # abstract
from letta.interface import CLIInterface  # for printing to terminal
from letta.log import get_logger
from letta.metadata import MetadataStore
from letta.o1_agent import O1Agent
from letta.offline_memory_agent import OfflineMemoryAgent
from letta.orm import Base
from letta.orm.errors import NoResultFound
from letta.prompts import gpt_system
from letta.providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
    GroqProvider,
    LettaProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    TogetherProvider,
    VLLMChatCompletionsProvider,
    VLLMCompletionsProvider,
)
from letta.schemas.agent import (
    AgentState,
    AgentType,
    CreateAgent,
    PersistedAgentState,
    UpdateAgentState,
)
from letta.schemas.api_key import APIKey, APIKeyCreate
from letta.schemas.block import Block, BlockUpdate
from letta.schemas.embedding_config import EmbeddingConfig

# openai schemas
from letta.schemas.enums import JobStatus
from letta.schemas.job import Job, JobUpdate
from letta.schemas.letta_message import FunctionReturn, LettaMessage
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import (
    ArchivalMemorySummary,
    ContextWindowOverview,
    Memory,
    RecallMemorySummary,
)
from letta.schemas.message import Message, MessageCreate, MessageRole, MessageUpdate
from letta.schemas.organization import Organization
from letta.schemas.passage import Passage
from letta.schemas.source import Source
from letta.schemas.tool import Tool, ToolCreate
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.services.agents_tags_manager import AgentsTagsManager
from letta.services.block_manager import BlockManager
from letta.services.blocks_agents_manager import BlocksAgentsManager
from letta.services.job_manager import JobManager
from letta.services.message_manager import MessageManager
from letta.services.organization_manager import OrganizationManager
from letta.services.per_agent_lock_manager import PerAgentLockManager
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.source_manager import SourceManager
from letta.services.tool_execution_sandbox import ToolExecutionSandbox
from letta.services.tool_manager import ToolManager
from letta.services.tools_agents_manager import ToolsAgentsManager
from letta.services.user_manager import UserManager
from letta.utils import create_random_username, get_utc_time, json_dumps, json_loads

logger = get_logger(__name__)


class Server(object):
    """Abstract server class that supports multi-agent multi-user"""

    @abstractmethod
    def list_agents(self, user_id: str) -> dict:
        """List all available agents to a user"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_messages(self, user_id: str, agent_id: str, start: int, count: int) -> list:
        """Paginated query of in-context messages in agent message queue"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_memory(self, user_id: str, agent_id: str) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_state(self, user_id: str, agent_id: str) -> dict:
        """Return the config of an agent"""
        raise NotImplementedError

    @abstractmethod
    def get_server_config(self, user_id: str) -> dict:
        """Return the base config"""
        raise NotImplementedError

    @abstractmethod
    def update_agent_core_memory(self, user_id: str, agent_id: str, new_memory_contents: dict) -> dict:
        """Update the agents core memory block, return the new state"""
        raise NotImplementedError

    @abstractmethod
    def create_agent(
        self,
        request: CreateAgent,
        actor: User,
        # interface
        interface: Union[AgentInterface, None] = None,
    ) -> AgentState:
        """Create a new agent using a config"""
        raise NotImplementedError

    @abstractmethod
    def user_message(self, user_id: str, agent_id: str, message: str) -> None:
        """Process a message from the user, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def system_message(self, user_id: str, agent_id: str, message: str) -> None:
        """Process a message from the system, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def send_messages(self, user_id: str, agent_id: str, messages: Union[MessageCreate, List[Message]]) -> None:
        """Send a list of messages to the agent"""
        raise NotImplementedError

    @abstractmethod
    def run_command(self, user_id: str, agent_id: str, command: str) -> Union[str, None]:
        """Run a command on the agent, e.g. /memory

        May return a string with a message generated by the command
        """
        raise NotImplementedError


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from letta.config import LettaConfig

# NOTE: hack to see if single session management works
from letta.settings import model_settings, settings, tool_settings

config = LettaConfig.load()

attach_base()

if settings.letta_pg_uri_no_default:
    config.recall_storage_type = "postgres"
    config.recall_storage_uri = settings.letta_pg_uri_no_default
    config.archival_storage_type = "postgres"
    config.archival_storage_uri = settings.letta_pg_uri_no_default

    # create engine
    engine = create_engine(settings.letta_pg_uri)
else:
    # TODO: don't rely on config storage
    engine = create_engine("sqlite:///" + os.path.join(config.recall_storage_path, "sqlite.db"))

    Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from contextlib import contextmanager

db_context = contextmanager(get_db)


class SyncServer(Server):
    """Simple single-threaded / blocking server process"""

    def __init__(
        self,
        chaining: bool = True,
        max_chaining_steps: Optional[bool] = None,
        default_interface_factory: Callable[[], AgentInterface] = lambda: CLIInterface(),
        init_with_default_org_and_user: bool = True,
        # default_interface: AgentInterface = CLIInterface(),
        # default_persistence_manager_cls: PersistenceManager = LocalStateManager,
        # auth_mode: str = "none",  # "none, "jwt", "external"
    ):
        """Server process holds in-memory agents that are being run"""

        # List of {'user_id': user_id, 'agent_id': agent_id, 'agent': agent_obj} dicts
        self.active_agents = []

        # chaining = whether or not to run again if request_heartbeat=true
        self.chaining = chaining

        # if chaining == true, what's the max number of times we'll chain before yielding?
        # none = no limit, can go on forever
        self.max_chaining_steps = max_chaining_steps

        # The default interface that will get assigned to agents ON LOAD
        self.default_interface_factory = default_interface_factory

        self.credentials = LettaCredentials.load()

        # Locks
        self.send_message_lock = Lock()

        # Initialize the metadata store
        config = LettaConfig.load()
        if settings.letta_pg_uri_no_default:
            config.recall_storage_type = "postgres"
            config.recall_storage_uri = settings.letta_pg_uri_no_default
            config.archival_storage_type = "postgres"
            config.archival_storage_uri = settings.letta_pg_uri_no_default
        config.save()
        self.config = config
        self.ms = MetadataStore(self.config)

        # Managers that interface with data models
        self.organization_manager = OrganizationManager()
        self.user_manager = UserManager()
        self.tool_manager = ToolManager()
        self.block_manager = BlockManager()
        self.source_manager = SourceManager()
        self.agents_tags_manager = AgentsTagsManager()
        self.sandbox_config_manager = SandboxConfigManager(tool_settings)
        self.blocks_agents_manager = BlocksAgentsManager()
        self.message_manager = MessageManager()
        self.tools_agents_manager = ToolsAgentsManager()
        self.job_manager = JobManager()

        # Managers that interface with parallelism
        self.per_agent_lock_manager = PerAgentLockManager()

        # Make default user and org
        if init_with_default_org_and_user:
            self.default_org = self.organization_manager.create_default_organization()
            self.default_user = self.user_manager.create_default_user()
            self.block_manager.add_default_blocks(actor=self.default_user)
            self.tool_manager.add_base_tools(actor=self.default_user)

            # If there is a default org/user
            # This logic may have to change in the future
            if settings.load_default_external_tools:
                self.add_default_external_tools(actor=self.default_user)

        # collect providers (always has Letta as a default)
        self._enabled_providers: List[Provider] = [LettaProvider()]
        if model_settings.openai_api_key:
            self._enabled_providers.append(
                OpenAIProvider(
                    api_key=model_settings.openai_api_key,
                    base_url=model_settings.openai_api_base,
                )
            )
        if model_settings.anthropic_api_key:
            self._enabled_providers.append(
                AnthropicProvider(
                    api_key=model_settings.anthropic_api_key,
                )
            )
        if model_settings.ollama_base_url:
            self._enabled_providers.append(
                OllamaProvider(
                    base_url=model_settings.ollama_base_url,
                    api_key=None,
                    default_prompt_formatter=model_settings.default_prompt_formatter,
                )
            )
        if model_settings.gemini_api_key:
            self._enabled_providers.append(
                GoogleAIProvider(
                    api_key=model_settings.gemini_api_key,
                )
            )
        if model_settings.azure_api_key and model_settings.azure_base_url:
            assert model_settings.azure_api_version, "AZURE_API_VERSION is required"
            self._enabled_providers.append(
                AzureProvider(
                    api_key=model_settings.azure_api_key,
                    base_url=model_settings.azure_base_url,
                    api_version=model_settings.azure_api_version,
                )
            )
        if model_settings.groq_api_key:
            self._enabled_providers.append(
                GroqProvider(
                    api_key=model_settings.groq_api_key,
                )
            )
        if model_settings.together_api_key:
            self._enabled_providers.append(
                TogetherProvider(
                    api_key=model_settings.together_api_key,
                    default_prompt_formatter=model_settings.default_prompt_formatter,
                )
            )
        if model_settings.vllm_api_base:
            # vLLM exposes both a /chat/completions and a /completions endpoint
            self._enabled_providers.append(
                VLLMCompletionsProvider(
                    base_url=model_settings.vllm_api_base,
                    default_prompt_formatter=model_settings.default_prompt_formatter,
                )
            )
            # NOTE: to use the /chat/completions endpoint, you need to specify extra flags on vLLM startup
            # see: https://docs.vllm.ai/en/latest/getting_started/examples/openai_chat_completion_client_with_tools.html
            # e.g. "... --enable-auto-tool-choice --tool-call-parser hermes"
            self._enabled_providers.append(
                VLLMChatCompletionsProvider(
                    base_url=model_settings.vllm_api_base,
                )
            )

    def save_agents(self):
        """Saves all the agents that are in the in-memory object store"""
        for agent_d in self.active_agents:
            try:
                save_agent(agent_d["agent"], self.ms)
                logger.info(f"Saved agent {agent_d['agent_id']}")
            except Exception as e:
                logger.exception(f"Error occurred while trying to save agent {agent_d['agent_id']}:\n{e}")

    def _get_agent(self, user_id: str, agent_id: str) -> Union[Agent, None]:
        """Get the agent object from the in-memory object store"""
        for d in self.active_agents:
            if d["user_id"] == str(user_id) and d["agent_id"] == str(agent_id):
                return d["agent"]
        return None

    def _add_agent(self, user_id: str, agent_id: str, agent_obj: Agent) -> None:
        """Put an agent object inside the in-memory object store"""
        # Make sure the agent doesn't already exist
        if self._get_agent(user_id=user_id, agent_id=agent_id) is not None:
            # Can be triggered on concucrent request, so don't throw a full error
            logger.exception(f"Agent (user={user_id}, agent={agent_id}) is already loaded")
            return
        # Add Agent instance to the in-memory list
        self.active_agents.append(
            {
                "user_id": str(user_id),
                "agent_id": str(agent_id),
                "agent": agent_obj,
            }
        )

    def initialize_agent(self, agent_id, interface: Union[AgentInterface, None] = None, initial_message_sequence=None) -> Agent:
        """Initialize an agent from the database"""
        agent_state = self.get_agent(agent_id=agent_id)
        actor = self.user_manager.get_user_by_id(user_id=agent_state.user_id)

        interface = interface or self.default_interface_factory()
        if agent_state.agent_type == AgentType.memgpt_agent:
            agent = Agent(agent_state=agent_state, interface=interface, user=actor, initial_message_sequence=initial_message_sequence)
        else:
            assert initial_message_sequence is None, f"Initial message sequence is not supported for O1Agents"
            agent = O1Agent(agent_state=agent_state, interface=interface, user=actor)

        # Persist to agent
        save_agent(agent, self.ms)
        return agent

    def load_agent(self, agent_id: str, interface: Union[AgentInterface, None] = None) -> Agent:
        """Updated method to load agents from persisted storage"""
        agent_lock = self.per_agent_lock_manager.get_lock(agent_id)
        with agent_lock:
            agent_state = self.get_agent(agent_id=agent_id)
            if agent_state is None:
                raise LettaAgentNotFoundError(f"Agent (agent_id={agent_id}) does not exist")
            elif agent_state.user_id is None:
                raise ValueError(f"Agent (agent_id={agent_id}) does not have a user_id")
            actor = self.user_manager.get_user_by_id(user_id=agent_state.user_id)

            interface = interface or self.default_interface_factory()
            if agent_state.agent_type == AgentType.memgpt_agent:
                agent = Agent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.o1_agent:
                agent = O1Agent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.offline_memory_agent:
                agent = OfflineMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.chat_only_agent:
                agent = ChatOnlyAgent(agent_state=agent_state, interface=interface, user=actor)
            else:
                raise ValueError(f"Invalid agent type {agent_state.agent_type}")

            # Rebuild the system prompt - may be linked to new blocks now
            agent.rebuild_system_prompt()

            # Persist to agent
            save_agent(agent, self.ms)
            return agent

    def _step(
        self,
        user_id: str,
        agent_id: str,
        input_messages: Union[Message, List[Message]],
        interface: Union[AgentInterface, None] = None,  # needed to getting responses
        # timestamp: Optional[datetime],
    ) -> LettaUsageStatistics:
        """Send the input message through the agent"""

        # Input validation
        if isinstance(input_messages, Message):
            input_messages = [input_messages]
        if not all(isinstance(m, Message) for m in input_messages):
            raise ValueError(f"messages should be a Message or a list of Message, got {type(input_messages)}")

        logger.debug(f"Got input messages: {input_messages}")
        letta_agent = None
        try:

            # Get the agent object (loaded in memory)
            # letta_agent = self._get_or_load_agent(agent_id=agent_id)
            letta_agent = self.load_agent(agent_id=agent_id, interface=interface)
            if letta_agent is None:
                raise KeyError(f"Agent (user={user_id}, agent={agent_id}) is not loaded")

            # Determine whether or not to token stream based on the capability of the interface
            token_streaming = letta_agent.interface.streaming_mode if hasattr(letta_agent.interface, "streaming_mode") else False

            logger.debug(f"Starting agent step")
            usage_stats = letta_agent.step(
                messages=input_messages,
                chaining=self.chaining,
                max_chaining_steps=self.max_chaining_steps,
                stream=token_streaming,
                ms=self.ms,
                skip_verify=True,
            )

            # save agent after step
            save_agent(letta_agent, self.ms)

        except Exception as e:
            logger.error(f"Error in server._step: {e}")
            print(traceback.print_exc())
            raise
        finally:
            logger.debug("Calling step_yield()")
            if letta_agent:
                letta_agent.interface.step_yield()

        return usage_stats

    def _command(self, user_id: str, agent_id: str, command: str) -> LettaUsageStatistics:
        """Process a CLI command"""

        logger.debug(f"Got command: {command}")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)
        usage = None

        if command.lower() == "exit":
            # exit not supported on server.py
            raise ValueError(command)

        elif command.lower() == "save" or command.lower() == "savechat":
            save_agent(letta_agent, self.ms)

        elif command.lower() == "attach":
            # Different from CLI, we extract the data source name from the command
            command = command.strip().split()
            try:
                data_source = int(command[1])
            except:
                raise ValueError(command)

            # attach data to agent from source
            source_connector = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)
            letta_agent.attach_source(data_source, source_connector, self.ms)

        elif command.lower() == "dump" or command.lower().startswith("dump "):
            # Check if there's an additional argument that's an integer
            command = command.strip().split()
            amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 0
            if amount == 0:
                letta_agent.interface.print_messages(letta_agent.messages, dump=True)
            else:
                letta_agent.interface.print_messages(letta_agent.messages[-min(amount, len(letta_agent.messages)) :], dump=True)

        elif command.lower() == "dumpraw":
            letta_agent.interface.print_messages_raw(letta_agent.messages)

        elif command.lower() == "memory":
            ret_str = f"\nDumping memory contents:\n" + f"\n{str(letta_agent.agent_state.memory)}" + f"\n{str(letta_agent.archival_memory)}"
            return ret_str

        elif command.lower() == "pop" or command.lower().startswith("pop "):
            # Check if there's an additional argument that's an integer
            command = command.strip().split()
            pop_amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
            n_messages = len(letta_agent.messages)
            MIN_MESSAGES = 2
            if n_messages <= MIN_MESSAGES:
                logger.debug(f"Agent only has {n_messages} messages in stack, none left to pop")
            elif n_messages - pop_amount < MIN_MESSAGES:
                logger.debug(f"Agent only has {n_messages} messages in stack, cannot pop more than {n_messages - MIN_MESSAGES}")
            else:
                logger.debug(f"Popping last {pop_amount} messages from stack")
                for _ in range(min(pop_amount, len(letta_agent.messages))):
                    letta_agent.messages.pop()

        elif command.lower() == "retry":
            # TODO this needs to also modify the persistence manager
            logger.debug(f"Retrying for another answer")
            while len(letta_agent.messages) > 0:
                if letta_agent.messages[-1].get("role") == "user":
                    # we want to pop up to the last user message and send it again
                    letta_agent.messages[-1].get("content")
                    letta_agent.messages.pop()
                    break
                letta_agent.messages.pop()

        elif command.lower() == "rethink" or command.lower().startswith("rethink "):
            # TODO this needs to also modify the persistence manager
            if len(command) < len("rethink "):
                logger.warning("Missing text after the command")
            else:
                for x in range(len(letta_agent.messages) - 1, 0, -1):
                    if letta_agent.messages[x].get("role") == "assistant":
                        text = command[len("rethink ") :].strip()
                        letta_agent.messages[x].update({"content": text})
                        break

        elif command.lower() == "rewrite" or command.lower().startswith("rewrite "):
            # TODO this needs to also modify the persistence manager
            if len(command) < len("rewrite "):
                logger.warning("Missing text after the command")
            else:
                for x in range(len(letta_agent.messages) - 1, 0, -1):
                    if letta_agent.messages[x].get("role") == "assistant":
                        text = command[len("rewrite ") :].strip()
                        args = json_loads(letta_agent.messages[x].get("function_call").get("arguments"))
                        args["message"] = text
                        letta_agent.messages[x].get("function_call").update({"arguments": json_dumps(args)})
                        break

        # No skip options
        elif command.lower() == "wipe":
            # exit not supported on server.py
            raise ValueError(command)

        elif command.lower() == "heartbeat":
            input_message = system.get_heartbeat()
            usage = self._step(user_id=user_id, agent_id=agent_id, input_message=input_message)

        elif command.lower() == "memorywarning":
            input_message = system.get_token_limit_warning()
            usage = self._step(user_id=user_id, agent_id=agent_id, input_message=input_message)

        if not usage:
            usage = LettaUsageStatistics()

        return usage

    def user_message(
        self,
        user_id: str,
        agent_id: str,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ) -> LettaUsageStatistics:
        """Process an incoming user message and feed it through the Letta agent"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_user_message = system.package_user_message(
                user_message=message,
                time=timestamp.isoformat() if timestamp else None,
            )

            # NOTE: eventually deprecate and only allow passing Message types
            # Convert to a Message object
            if timestamp:
                message = Message(
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                )

        # Run the agent state forward
        usage = self._step(user_id=user_id, agent_id=agent_id, input_messages=message)
        return usage

    def system_message(
        self,
        user_id: str,
        agent_id: str,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ) -> LettaUsageStatistics:
        """Process an incoming system message and feed it through the Letta agent"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_system_message = system.package_system_message(system_message=message)

            # NOTE: eventually deprecate and only allow passing Message types
            # Convert to a Message object

            if timestamp:
                message = Message(
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                )

        if isinstance(message, Message):
            # Can't have a null text field
            if message.text is None or len(message.text) == 0:
                raise ValueError(f"Invalid input: '{message.text}'")
            # If the input begins with a command prefix, reject
            elif message.text.startswith("/"):
                raise ValueError(f"Invalid input: '{message.text}'")

        else:
            raise TypeError(f"Invalid input: '{message}' - type {type(message)}")

        if timestamp:
            # Override the timestamp with what the caller provided
            message.created_at = timestamp

        # Run the agent state forward
        return self._step(user_id=user_id, agent_id=agent_id, input_messages=message)

    def send_messages(
        self,
        user_id: str,
        agent_id: str,
        messages: Union[List[MessageCreate], List[Message]],
        # whether or not to wrap user and system message as MemGPT-style stringified JSON
        wrap_user_message: bool = True,
        wrap_system_message: bool = True,
        interface: Union[AgentInterface, None] = None,  # needed to getting responses
    ) -> LettaUsageStatistics:
        """Send a list of messages to the agent

        If the messages are of type MessageCreate, we need to turn them into
        Message objects first before sending them through step.

        Otherwise, we can pass them in directly.
        """
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        message_objects: List[Message] = []

        if all(isinstance(m, MessageCreate) for m in messages):
            for message in messages:
                assert isinstance(message, MessageCreate)

                # If wrapping is eanbled, wrap with metadata before placing content inside the Message object
                if message.role == MessageRole.user and wrap_user_message:
                    message.text = system.package_user_message(user_message=message.text)
                elif message.role == MessageRole.system and wrap_system_message:
                    message.text = system.package_system_message(system_message=message.text)
                else:
                    raise ValueError(f"Invalid message role: {message.role}")

                # Create the Message object
                message_objects.append(
                    Message(
                        agent_id=agent_id,
                        role=message.role,
                        text=message.text,
                        name=message.name,
                        # assigned later?
                        model=None,
                        # irrelevant
                        tool_calls=None,
                        tool_call_id=None,
                    )
                )

        elif all(isinstance(m, Message) for m in messages):
            for message in messages:
                assert isinstance(message, Message)
                message_objects.append(message)

        else:
            raise ValueError(f"All messages must be of type Message or MessageCreate, got {[type(message) for message in messages]}")

        # Run the agent state forward
        return self._step(user_id=user_id, agent_id=agent_id, input_messages=message_objects, interface=interface)

    # @LockingServer.agent_lock_decorator
    def run_command(self, user_id: str, agent_id: str, command: str) -> LettaUsageStatistics:
        """Run a command on the agent"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # If the input begins with a command prefix, attempt to process it as a command
        if command.startswith("/"):
            if len(command) > 1:
                command = command[1:]  # strip the prefix
        return self._command(user_id=user_id, agent_id=agent_id, command=command)

    def create_agent(
        self,
        request: CreateAgent,
        actor: User,
        # interface
        interface: Union[AgentInterface, None] = None,
    ) -> AgentState:
        """Create a new agent using a config"""
        user_id = actor.id
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        if interface is None:
            interface = self.default_interface_factory()

        # create agent name
        if request.name is None:
            request.name = create_random_username()

        if request.agent_type is None:
            request.agent_type = AgentType.memgpt_agent

        # system debug
        if request.system is None:
            # TODO: don't hardcode
            if request.agent_type == AgentType.memgpt_agent:
                request.system = gpt_system.get_system_text("memgpt_chat")
            elif request.agent_type == AgentType.o1_agent:
                request.system = gpt_system.get_system_text("memgpt_modified_o1")
            elif request.agent_type == AgentType.offline_memory_agent:
                request.system = gpt_system.get_system_text("memgpt_offline_memory")
            elif request.agent_type == AgentType.chat_only_agent:
                request.system = gpt_system.get_system_text("memgpt_convo_only")
            else:
                raise ValueError(f"Invalid agent type: {request.agent_type}")

        # create blocks (note: cannot be linked into the agent_id is created)
        blocks = []
        for create_block in request.memory_blocks:
            block = self.block_manager.create_or_update_block(Block(**create_block.model_dump()), actor=actor)
            blocks.append(block)

        # get tools + only add if they exist
        tool_objs = []
        if request.tools:
            for tool_name in request.tools:
                tool_obj = self.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor)
                if tool_obj:
                    tool_objs.append(tool_obj)
                else:
                    warnings.warn(f"Attempted to add a nonexistent tool {tool_name} to agent {request.name}, skipping.")
        # reset the request.tools to only valid tools
        request.tools = [t.name for t in tool_objs]

        # get the user
        logger.debug(f"Attempting to find user: {user_id}")
        user = self.user_manager.get_user_by_id(user_id=user_id)
        if not user:
            raise ValueError(f"cannot find user with associated client id: {user_id}")

        # created and persist the agent state in the DB
        agent_state = PersistedAgentState(
            name=request.name,
            user_id=user_id,
            tool_names=request.tools if request.tools else [],
            tool_rules=request.tool_rules,
            agent_type=request.agent_type or AgentType.memgpt_agent,
            llm_config=request.llm_config,
            embedding_config=request.embedding_config,
            system=request.system,
            # other metadata
            description=request.description,
            metadata_=request.metadata_,
        )
        # TODO: move this to agent ORM
        # this saves the agent ID and state into the DB
        self.ms.create_agent(agent_state)

        # create the agent object
        if request.initial_message_sequence:
            # init_messages = [Message(user_id=user_id, agent_id=agent_state.id, role=message.role, text=message.text) for message in request.initial_message_sequence]
            init_messages = []
            for message in request.initial_message_sequence:

                if message.role == MessageRole.user:
                    packed_message = system.package_user_message(
                        user_message=message.text,
                    )
                elif message.role == MessageRole.system:
                    packed_message = system.package_system_message(
                        system_message=message.text,
                    )
                else:
                    raise ValueError(f"Invalid message role: {message.role}")

                init_messages.append(Message(role=message.role, text=packed_message, agent_id=agent_state.id))
            # init_messages = [Message.dict_to_message(user_id=user_id, agent_id=agent_state.id, openai_message_dict=message.model_dump()) for message in request.initial_message_sequence]
        else:
            init_messages = None

        # initialize the agent (generates initial message list with system prompt)
        self.initialize_agent(agent_id=agent_state.id, interface=interface, initial_message_sequence=init_messages)

        # Note: mappings (e.g. tags, blocks) are created after the agent is persisted
        # TODO: add source mappings here as well

        # create the tags
        if request.tags:
            for tag in request.tags:
                self.agents_tags_manager.add_tag_to_agent(agent_id=agent_state.id, tag=tag, actor=actor)

        # create block mappins (now that agent is persisted)
        for block in blocks:
            # this links the created block to the agent
            self.blocks_agents_manager.add_block_to_agent(block_id=block.id, agent_id=agent_state.id, block_label=block.label)

        in_memory_agent_state = self.get_agent(agent_state.id)
        return in_memory_agent_state

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """
        Retrieve the full agent state from the DB.
        This gathers data accross multiple tables to provide the full state of an agent, which is passed into the `Agent` object for creation.
        """

        # get data persisted from the DB
        agent_state = self.ms.get_agent(agent_id=agent_id)
        if agent_state is None:
            # agent does not exist
            return None
        if agent_state.user_id is None:
            raise ValueError(f"Agent {agent_id} does not have a user_id")
        user = self.user_manager.get_user_by_id(user_id=agent_state.user_id)

        # construct the in-memory, full agent state - this gather data stored in different tables but that needs to be passed to `Agent`
        # we also return this data to the user to provide all the state related to an agent

        # get `Memory` object by getting the linked block IDs and fetching the blocks, then putting that into a `Memory` object
        # this is the "in memory" representation of the in-context memory
        block_ids = self.blocks_agents_manager.list_block_ids_for_agent(agent_id=agent_id)
        blocks = []
        for block_id in block_ids:
            block = self.block_manager.get_block_by_id(block_id=block_id, actor=user)
            assert block, f"Block with ID {block_id} does not exist"
            blocks.append(block)
        memory = Memory(blocks=blocks)

        # get `Tool` objects
        tools = [self.tool_manager.get_tool_by_name(tool_name=tool_name, actor=user) for tool_name in agent_state.tool_names]

        # get `Source` objects
        sources = self.list_attached_sources(agent_id=agent_id)

        # get the tags
        tags = self.agents_tags_manager.get_tags_for_agent(agent_id=agent_id, actor=user)

        # return the full agent state - this contains all data needed to recreate the agent
        return AgentState(**agent_state.model_dump(), memory=memory, tools=tools, sources=sources, tags=tags)

    def update_agent(
        self,
        request: UpdateAgentState,
        actor: User,
    ) -> AgentState:
        """Update the agents core memory block, return the new state"""
        try:
            self.user_manager.get_user_by_id(user_id=actor.id)
        except Exception:
            raise ValueError(f"User user_id={actor.id} does not exist")

        if self.ms.get_agent(agent_id=request.id) is None:
            raise ValueError(f"Agent agent_id={request.id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=request.id)

        # update the system prompt
        if request.system:
            letta_agent.update_system_prompt(request.system)

        # update in-context messages
        if request.message_ids:
            # This means the user is trying to change what messages are in the message buffer
            # Internally this requires (1) pulling from recall,
            # then (2) setting the attributes ._messages and .state.message_ids
            letta_agent.set_message_buffer(message_ids=request.message_ids)

        # tools
        if request.tool_names:
            # Replace tools and also re-link

            # (1) get tools + make sure they exist
            # Current and target tools as sets of tool names
            current_tools = set(letta_agent.agent_state.tool_names)
            target_tools = set(request.tool_names)

            # Calculate tools to add and remove
            tools_to_add = target_tools - current_tools
            tools_to_remove = current_tools - target_tools

            # Fetch tool objects for those to add and remove
            tools_to_add = [self.tool_manager.get_tool_by_name(tool_name=tool, actor=actor) for tool in tools_to_add]
            tools_to_remove = [self.tool_manager.get_tool_by_name(tool_name=tool, actor=actor) for tool in tools_to_remove]

            # update agent tool list
            for tool in tools_to_remove:
                self.remove_tool_from_agent(agent_id=request.id, tool_id=tool.id, user_id=actor.id)
            for tool in tools_to_add:
                self.add_tool_to_agent(agent_id=request.id, tool_id=tool.id, user_id=actor.id)

            # reload agent
            letta_agent = self.load_agent(agent_id=request.id)

        # configs
        if request.llm_config:
            letta_agent.agent_state.llm_config = request.llm_config
        if request.embedding_config:
            letta_agent.agent_state.embedding_config = request.embedding_config

        # other minor updates
        if request.name:
            letta_agent.agent_state.name = request.name
        if request.metadata_:
            letta_agent.agent_state.metadata_ = request.metadata_

        # Manage tag state
        if request.tags is not None:
            current_tags = set(self.agents_tags_manager.get_tags_for_agent(agent_id=letta_agent.agent_state.id, actor=actor))
            target_tags = set(request.tags)

            tags_to_add = target_tags - current_tags
            tags_to_remove = current_tags - target_tags

            for tag in tags_to_add:
                self.agents_tags_manager.add_tag_to_agent(agent_id=letta_agent.agent_state.id, tag=tag, actor=actor)
            for tag in tags_to_remove:
                self.agents_tags_manager.delete_tag_from_agent(agent_id=letta_agent.agent_state.id, tag=tag, actor=actor)

        # save the agent
        save_agent(letta_agent, self.ms)
        # TODO: probably reload the agent somehow?
        return letta_agent.agent_state

    def get_tools_from_agent(self, agent_id: str, user_id: Optional[str]) -> List[Tool]:
        """Get tools from an existing agent"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)
        return letta_agent.agent_state.tools

    def add_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str,
        user_id: str,
    ):
        """Add tools from an existing agent"""
        try:
            user = self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise ValueError(f"User user_id={user_id} does not exist")

        if self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # Get all the tool objects from the request
        tool_objs = []
        tool_obj = self.tool_manager.get_tool_by_id(tool_id=tool_id, actor=user)
        assert tool_obj, f"Tool with id={tool_id} does not exist"
        tool_objs.append(tool_obj)

        for tool in letta_agent.agent_state.tools:
            tool_obj = self.tool_manager.get_tool_by_id(tool_id=tool.id, actor=user)
            assert tool_obj, f"Tool with id={tool.id} does not exist"

            # If it's not the already added tool
            if tool_obj.id != tool_id:
                tool_objs.append(tool_obj)

        # replace the list of tool names ("ids") inside the agent state
        letta_agent.agent_state.tool_names = [tool.name for tool in tool_objs]

        # then attempt to link the tools modules
        letta_agent.link_tools(tool_objs)

        # save the agent
        save_agent(letta_agent, self.ms)
        return letta_agent.agent_state

    def remove_tool_from_agent(
        self,
        agent_id: str,
        tool_id: str,
        user_id: str,
    ):
        """Remove tools from an existing agent"""
        try:
            user = self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise ValueError(f"User user_id={user_id} does not exist")

        if self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # Get all the tool_objs
        tool_objs = []
        for tool in letta_agent.agent_state.tools:
            tool_obj = self.tool_manager.get_tool_by_id(tool_id=tool.id, actor=user)
            assert tool_obj, f"Tool with id={tool.id} does not exist"

            # If it's not the tool we want to remove
            if tool_obj.id != tool_id:
                tool_objs.append(tool_obj)

        # replace the list of tool names ("ids") inside the agent state
        letta_agent.agent_state.tool_names = [tool.name for tool in tool_objs]

        # then attempt to link the tools modules
        letta_agent.link_tools(tool_objs)

        # save the agent
        save_agent(letta_agent, self.ms)
        return letta_agent.agent_state

    def get_agent_state(self, user_id: str, agent_id: str) -> AgentState:
        # TODO: duplicate, remove
        return self.get_agent(agent_id=agent_id)

    def list_agents(self, user_id: str, tags: Optional[List[str]] = None) -> List[AgentState]:
        """List all available agents to a user"""
        user = self.user_manager.get_user_by_id(user_id=user_id)

        if tags is None:
            agents_states = self.ms.list_agents(user_id=user_id)
            agent_ids = [agent.id for agent in agents_states]
        else:
            agent_ids = []
            for tag in tags:
                agent_ids += self.agents_tags_manager.get_agents_by_tag(tag=tag, actor=user)

        return [self.get_agent(agent_id=agent_id) for agent_id in agent_ids]

    # convert name->id

    def get_agent_id(self, name: str, user_id: str):
        agent_state = self.ms.get_agent(agent_name=name, user_id=user_id)
        if not agent_state:
            return None
        return agent_state.id

    def get_source(self, source_id: str, user_id: str) -> Source:
        existing_source = self.ms.get_source(source_id=source_id, user_id=user_id)
        if not existing_source:
            raise ValueError("Source does not exist")
        return existing_source

    def get_source_id(self, source_name: str, user_id: str) -> str:
        existing_source = self.ms.get_source(source_name=source_name, user_id=user_id)
        if not existing_source:
            raise ValueError("Source does not exist")
        return existing_source.id

    def get_agent_memory(self, agent_id: str) -> Memory:
        """Return the memory of an agent (core memory)"""
        agent = self.load_agent(agent_id=agent_id)
        return agent.agent_state.memory

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        agent = self.load_agent(agent_id=agent_id)
        return ArchivalMemorySummary(size=len(agent.archival_memory))

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        agent = self.load_agent(agent_id=agent_id)
        return RecallMemorySummary(size=len(agent.message_manager))

    def get_in_context_message_ids(self, agent_id: str) -> List[str]:
        """Get the message ids of the in-context messages in the agent's memory"""
        # Get the agent object (loaded in memory)
        agent = self.load_agent(agent_id=agent_id)
        return [m.id for m in agent._messages]

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """Get the in-context messages in the agent's memory"""
        # Get the agent object (loaded in memory)
        agent = self.load_agent(agent_id=agent_id)
        return agent._messages

    def get_agent_message(self, agent_id: str, message_id: str) -> Message:
        """Get a single message from the agent's memory"""
        # Get the agent object (loaded in memory)
        agent = self.load_agent(agent_id=agent_id)
        message = agent.message_manager.get_message_by_id(id=message_id, actor=self.default_user)
        return message

    def get_agent_messages(
        self,
        agent_id: str,
        start: int,
        count: int,
    ) -> Union[List[Message], List[LettaMessage]]:
        """Paginated query of all messages in agent message queue"""
        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        if start < 0 or count < 0:
            raise ValueError("Start and count values should be non-negative")

        if start + count < len(letta_agent._messages):  # messages can be returned from whats in memory
            # Reverse the list to make it in reverse chronological order
            reversed_messages = letta_agent._messages[::-1]
            # Check if start is within the range of the list
            if start >= len(reversed_messages):
                raise IndexError("Start index is out of range")

            # Calculate the end index, ensuring it does not exceed the list length
            end_index = min(start + count, len(reversed_messages))

            # Slice the list for pagination
            messages = reversed_messages[start:end_index]

        else:
            # need to access persistence manager for additional messages

            # get messages using message manager
            page = letta_agent.message_manager.list_user_messages_for_agent(
                agent_id=agent_id,
                actor=self.default_user,
                cursor=start,
                limit=count,
            )

            messages = page
            assert all(isinstance(m, Message) for m in messages)

            ## Convert to json
            ## Add a tag indicating in-context or not
            # json_messages = [record.to_json() for record in messages]
            # in_context_message_ids = [str(m.id) for m in letta_agent._messages]
            # for d in json_messages:
            #    d["in_context"] = True if str(d["id"]) in in_context_message_ids else False

        return messages

    def get_agent_archival(self, user_id: str, agent_id: str, start: int, count: int) -> List[Passage]:
        """Paginated query of all messages in agent archival memory"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # iterate over records
        db_iterator = letta_agent.archival_memory.storage.get_all_paginated(page_size=count, offset=start)

        # get a single page of messages
        page = next(db_iterator, [])
        return page

    def get_agent_archival_cursor(
        self,
        user_id: str,
        agent_id: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Optional[int] = 100,
        order_by: Optional[str] = "created_at",
        reverse: Optional[bool] = False,
    ) -> List[Passage]:
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise LettaUserNotFoundError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise LettaAgentNotFoundError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # iterate over recorde
        cursor, records = letta_agent.archival_memory.storage.get_all_cursor(
            after=after, before=before, limit=limit, order_by=order_by, reverse=reverse
        )
        return records

    def insert_archival_memory(self, user_id: str, agent_id: str, memory_contents: str) -> List[Passage]:
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # Insert into archival memory
        passage_ids = letta_agent.archival_memory.insert(memory_string=memory_contents, return_ids=True)

        # Update the agent
        # TODO: should this update the system prompt?
        save_agent(letta_agent, self.ms)

        # TODO: this is gross, fix
        return [letta_agent.archival_memory.storage.get(id=passage_id) for passage_id in passage_ids]

    def delete_archival_memory(self, user_id: str, agent_id: str, memory_id: str):
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # TODO: should return a passage

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # Delete by ID
        # TODO check if it exists first, and throw error if not
        letta_agent.archival_memory.storage.delete({"id": memory_id})

        # TODO: return archival memory

    def get_agent_recall_cursor(
        self,
        user_id: str,
        agent_id: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = 100,
        reverse: Optional[bool] = False,
        return_message_object: bool = True,
        assistant_message_tool_name: str = constants.DEFAULT_MESSAGE_TOOL,
        assistant_message_tool_kwarg: str = constants.DEFAULT_MESSAGE_TOOL_KWARG,
    ) -> Union[List[Message], List[LettaMessage]]:
        actor = self.user_manager.get_user_by_id(user_id=user_id)
        if actor is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        # iterate over records
        # TODO: Check "order_by", "order"
        records = letta_agent.message_manager.list_messages_for_agent(
            agent_id=agent_id,
            actor=actor,
            cursor=cursor,
            limit=limit,
        )

        assert all(isinstance(m, Message) for m in records)

        if not return_message_object:
            # If we're GETing messages in reverse, we need to reverse the inner list (generated by to_letta_message)
            records = [
                msg
                for m in records
                for msg in m.to_letta_message(
                    assistant_message_tool_name=assistant_message_tool_name,
                    assistant_message_tool_kwarg=assistant_message_tool_kwarg,
                )
            ]

        if reverse:
            records = records[::-1]

        return records

    def get_server_config(self, include_defaults: bool = False) -> dict:
        """Return the base config"""

        def clean_keys(config):
            config_copy = config.copy()
            for k, v in config.items():
                if k == "key" or "_key" in k:
                    config_copy[k] = server_utils.shorten_key_middle(v, chars_each_side=5)
            return config_copy

        # TODO: do we need a seperate server config?
        base_config = vars(self.config)
        clean_base_config = clean_keys(base_config)

        response = {"config": clean_base_config}

        if include_defaults:
            default_config = vars(LettaConfig())
            clean_default_config = clean_keys(default_config)
            response["defaults"] = clean_default_config

        return response

    def update_agent_core_memory(self, user_id: str, agent_id: str, label: str, value: str) -> Memory:
        """Update the value of a block in the agent's memory"""

        # get the block id
        block = self.get_agent_block_by_label(user_id=user_id, agent_id=agent_id, label=label)
        block_id = block.id

        # update the block
        self.block_manager.update_block(
            block_id=block_id, block_update=BlockUpdate(value=value), actor=self.user_manager.get_user_by_id(user_id=user_id)
        )

        # load agent
        letta_agent = self.load_agent(agent_id=agent_id)
        return letta_agent.agent_state.memory

    def rename_agent(self, user_id: str, agent_id: str, new_agent_name: str) -> PersistedAgentState:
        """Update the name of the agent in the database"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)

        current_name = letta_agent.agent_state.name
        if current_name == new_agent_name:
            raise ValueError(f"New name ({new_agent_name}) is the same as the current name")

        try:
            letta_agent.agent_state.name = new_agent_name
            self.ms.update_agent(agent=letta_agent.agent_state)
        except Exception as e:
            logger.exception(f"Failed to update agent name with:\n{str(e)}")
            raise ValueError(f"Failed to update agent name in database")

        assert isinstance(letta_agent.agent_state.id, str)
        return letta_agent.agent_state

    def delete_agent(self, user_id: str, agent_id: str):
        """Delete an agent in the database"""
        actor = self.user_manager.get_user_by_id(user_id=user_id)
        # TODO: REMOVE THIS ONCE WE MIGRATE AGENTMODEL TO ORM MODEL
        # TODO: EVENTUALLY WE GET AUTO-DELETES WHEN WE SPECIFY RELATIONSHIPS IN THE ORM
        self.agents_tags_manager.delete_all_tags_from_agent(agent_id=agent_id, actor=actor)
        self.blocks_agents_manager.remove_all_agent_blocks(agent_id=agent_id)

        # Verify that the agent exists and belongs to the org of the user
        agent_state = self.ms.get_agent(agent_id=agent_id, user_id=user_id)
        if agent_state is None:
            raise ValueError(f"Could not find agent_id={agent_id} under user_id={user_id}")

        # TODO: REMOVE THIS ONCE WE MIGRATE AGENTMODEL TO ORM MODEL
        messages = self.message_manager.list_messages_for_agent(agent_id=agent_state.id)
        for message in messages:
            self.message_manager.delete_message_by_id(message.id, actor=actor)

        # TODO: REMOVE THIS ONCE WE MIGRATE AGENTMODEL TO ORM
        try:
            agent_state_user = self.user_manager.get_user_by_id(user_id=agent_state.user_id)
            if agent_state_user.organization_id != actor.organization_id:
                raise ValueError(
                    f"Could not authorize agent_id={agent_id} with user_id={user_id} because of differing organizations; agent_id was created in {agent_state_user.organization_id} while user belongs to {actor.organization_id}. How did they get the agent id?"
                )
        except NoResultFound:
            logger.error(f"Agent with id {agent_state.id} has nonexistent user {agent_state.user_id}")

        # First, if the agent is in the in-memory cache we should remove it
        # List of {'user_id': user_id, 'agent_id': agent_id, 'agent': agent_obj} dicts
        try:
            self.active_agents = [d for d in self.active_agents if str(d["agent_id"]) != str(agent_id)]
        except Exception as e:
            logger.exception(f"Failed to delete agent {agent_id} from cache via ID with:\n{str(e)}")
            raise ValueError(f"Failed to delete agent {agent_id} from cache")

        # Next, attempt to delete it from the actual database
        try:
            self.ms.delete_agent(agent_id=agent_id, per_agent_lock_manager=self.per_agent_lock_manager)
        except Exception as e:
            logger.exception(f"Failed to delete agent {agent_id} via ID with:\n{str(e)}")
            raise ValueError(f"Failed to delete agent {agent_id} in database")

    def api_key_to_user(self, api_key: str) -> str:
        """Decode an API key to a user"""
        token = self.ms.get_api_key(api_key=api_key)
        user = self.user_manager.get_user_by_id(token.user_id)
        if user is None:
            raise HTTPException(status_code=403, detail="Invalid credentials")
        else:
            return user.id

    def create_api_key(self, request: APIKeyCreate) -> APIKey:  # TODO: add other fields
        """Create a new API key for a user"""
        if request.name is None:
            request.name = f"API Key {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        token = self.ms.create_api_key(user_id=request.user_id, name=request.name)
        return token

    def list_api_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user"""
        return self.ms.get_all_api_keys_for_user(user_id=user_id)

    def delete_api_key(self, api_key: str) -> APIKey:
        api_key_obj = self.ms.get_api_key(api_key=api_key)
        if api_key_obj is None:
            raise ValueError("API key does not exist")
        self.ms.delete_api_key(api_key=api_key)
        return api_key_obj

    def delete_source(self, source_id: str, actor: User):
        """Delete a data source"""
        self.source_manager.delete_source(source_id=source_id, actor=actor)

        # delete data from passage store
        passage_store = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=actor.id)
        passage_store.delete({"source_id": source_id})

        # TODO: delete data from agent passage stores (?)

    def load_file_to_source(self, source_id: str, file_path: str, job_id: str, actor: User) -> Job:

        # update job
        job = self.job_manager.get_job_by_id(job_id, actor=actor)
        job.status = JobStatus.running
        self.job_manager.update_job_by_id(job_id=job_id, job_update=JobUpdate(**job.model_dump()), actor=actor)

        # try:
        from letta.data_sources.connectors import DirectoryConnector

        source = self.source_manager.get_source_by_id(source_id=source_id)
        connector = DirectoryConnector(input_files=[file_path])
        num_passages, num_documents = self.load_data(user_id=source.created_by_id, source_name=source.name, connector=connector)

        # update job status
        job.status = JobStatus.completed
        job.metadata_["num_passages"] = num_passages
        job.metadata_["num_documents"] = num_documents
        self.job_manager.update_job_by_id(job_id=job_id, job_update=JobUpdate(**job.model_dump()), actor=actor)

        return job

    def load_data(
        self,
        user_id: str,
        connector: DataConnector,
        source_name: str,
    ) -> Tuple[int, int]:
        """Load data from a DataConnector into a source for a specified user_id"""
        # TODO: this should be implemented as a batch job or at least async, since it may take a long time

        # load data from a data source into the document store
        user = self.user_manager.get_user_by_id(user_id=user_id)
        source = self.source_manager.get_source_by_name(source_name=source_name, actor=user)
        if source is None:
            raise ValueError(f"Data source {source_name} does not exist for user {user_id}")

        # get the data connectors
        passage_store = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)

        # load data into the document store
        passage_count, document_count = load_data(connector, source, passage_store, self.source_manager, actor=user)
        return passage_count, document_count

    def attach_source_to_agent(
        self,
        user_id: str,
        agent_id: str,
        # source_id: str,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Source:
        # attach a data source to an agent
        user = self.user_manager.get_user_by_id(user_id=user_id)
        if source_id:
            data_source = self.source_manager.get_source_by_id(source_id=source_id, actor=user)
        elif source_name:
            data_source = self.source_manager.get_source_by_name(source_name=source_name, actor=user)
        else:
            raise ValueError(f"Need to provide at least source_id or source_name to find the source.")
        # get connection to data source storage
        source_connector = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)
        assert data_source, f"Data source with id={source_id} or name={source_name} does not exist"

        # load agent
        agent = self.load_agent(agent_id=agent_id)

        # attach source to agent
        agent.attach_source(data_source.id, source_connector, self.ms)

        return data_source

    def detach_source_from_agent(
        self,
        user_id: str,
        agent_id: str,
        # source_id: str,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Source:
        user = self.user_manager.get_user_by_id(user_id=user_id)
        if source_id:
            source = self.source_manager.get_source_by_id(source_id=source_id, actor=user)
        elif source_name:
            source = self.source_manager.get_source_by_name(source_name=source_name, actor=user)
        else:
            raise ValueError(f"Need to provide at least source_id or source_name to find the source.")
        source_id = source.id

        # delete all Passage objects with source_id==source_id from agent's archival memory
        agent = self.load_agent(agent_id=agent_id)
        archival_memory = agent.archival_memory
        archival_memory.storage.delete({"source_id": source_id})

        # delete agent-source mapping
        self.ms.detach_source(agent_id=agent_id, source_id=source_id)

        # return back source data
        return source

    def list_attached_sources(self, agent_id: str) -> List[Source]:
        # list all attached sources to an agent
        source_ids = self.ms.list_attached_source_ids(agent_id)

        return [self.source_manager.get_source_by_id(source_id=id) for id in source_ids]

    def list_data_source_passages(self, user_id: str, source_id: str) -> List[Passage]:
        warnings.warn("list_data_source_passages is not yet implemented, returning empty list.", category=UserWarning)
        return []

    def list_all_sources(self, actor: User) -> List[Source]:
        """List all sources (w/ extra metadata) belonging to a user"""

        sources = self.source_manager.list_sources(actor=actor)

        # Add extra metadata to the sources
        sources_with_metadata = []
        for source in sources:

            # count number of passages
            passage_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=actor.id)
            num_passages = passage_conn.size({"source_id": source.id})

            # TODO: add when files table implemented
            ## count number of files
            # document_conn = StorageConnector.get_storage_connector(TableType.FILES, self.config, user_id=user_id)
            # num_documents = document_conn.size({"data_source": source.name})
            num_documents = 0

            agent_ids = self.ms.list_attached_agents(source_id=source.id)
            # add the agent name information
            attached_agents = [
                {
                    "id": str(a_id),
                    "name": self.ms.get_agent(user_id=actor.id, agent_id=a_id).name,
                }
                for a_id in agent_ids
            ]

            # Overwrite metadata field, should be empty anyways
            source.metadata_ = dict(
                num_documents=num_documents,
                num_passages=num_passages,
                attached_agents=attached_agents,
            )

            sources_with_metadata.append(source)

        return sources_with_metadata

    def add_default_external_tools(self, actor: User) -> bool:
        """Add default langchain tools. Return true if successful, false otherwise."""
        success = True
        tool_creates = ToolCreate.load_default_langchain_tools()
        if tool_settings.composio_api_key:
            tool_creates += ToolCreate.load_default_composio_tools()
        for tool_create in tool_creates:
            try:
                self.tool_manager.create_or_update_tool(Tool(**tool_create.model_dump()), actor=actor)
            except Exception as e:
                warnings.warn(f"An error occurred while creating tool {tool_create}: {e}")
                warnings.warn(traceback.format_exc())
                success = False

        return success

    def get_agent_message(self, agent_id: str, message_id: str) -> Optional[Message]:
        """Get a single message from the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = self.load_agent(agent_id=agent_id)
        message = letta_agent.message_manager.get_message_by_id(id=message_id)
        save_agent(letta_agent, self.ms)
        return message

    def update_agent_message(self, agent_id: str, message_id: str, request: MessageUpdate) -> Message:
        """Update the details of a message associated with an agent"""

        # Get the current message
        letta_agent = self.load_agent(agent_id=agent_id)
        response = letta_agent.update_message(message_id=message_id, request=request)
        save_agent(letta_agent, self.ms)
        return response

    def rewrite_agent_message(self, agent_id: str, new_text: str) -> Message:

        # Get the current message
        letta_agent = self.load_agent(agent_id=agent_id)
        response = letta_agent.rewrite_message(new_text=new_text)
        save_agent(letta_agent, self.ms)
        return response

    def rethink_agent_message(self, agent_id: str, new_thought: str) -> Message:

        # Get the current message
        letta_agent = self.load_agent(agent_id=agent_id)
        response = letta_agent.rethink_message(new_thought=new_thought)
        save_agent(letta_agent, self.ms)
        return response

    def retry_agent_message(self, agent_id: str) -> List[Message]:

        # Get the current message
        letta_agent = self.load_agent(agent_id=agent_id)
        response = letta_agent.retry_message()
        save_agent(letta_agent, self.ms)
        return response

    def get_user_or_default(self, user_id: Optional[str]) -> User:
        """Get the user object for user_id if it exists, otherwise return the default user object"""
        if user_id is None:
            user_id = self.user_manager.DEFAULT_USER_ID

        try:
            return self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"User with id {user_id} not found")

    def get_organization_or_default(self, org_id: Optional[str]) -> Organization:
        """Get the organization object for org_id if it exists, otherwise return the default organization object"""
        if org_id is None:
            org_id = self.organization_manager.DEFAULT_ORG_ID

        try:
            return self.organization_manager.get_organization_by_id(org_id=org_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"Organization with id {org_id} not found")

    def list_llm_models(self) -> List[LLMConfig]:
        """List available models"""

        llm_models = []
        for provider in self._enabled_providers:
            llm_models.extend(provider.list_llm_models())
        return llm_models

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        """List available embedding models"""
        embedding_models = []
        for provider in self._enabled_providers:
            embedding_models.extend(provider.list_embedding_models())
        return embedding_models

    def add_llm_model(self, request: LLMConfig) -> LLMConfig:
        """Add a new LLM model"""

    def add_embedding_model(self, request: EmbeddingConfig) -> EmbeddingConfig:
        """Add a new embedding model"""

    def get_agent_context_window(
        self,
        user_id: str,
        agent_id: str,
    ) -> ContextWindowOverview:
        # Get the current message
        letta_agent = self.load_agent(agent_id=agent_id)
        return letta_agent.get_context_window()

    def link_block_to_agent_memory(self, user_id: str, agent_id: str, block_id: str) -> Memory:
        """Link a block to an agent's memory"""
        block = self.block_manager.get_block_by_id(block_id=block_id, actor=self.user_manager.get_user_by_id(user_id=user_id))
        if block is None:
            raise ValueError(f"Block with id {block_id} not found")
        self.blocks_agents_manager.add_block_to_agent(agent_id, block_id, block_label=block.label)

        # get agent memory
        memory = self.get_agent(agent_id=agent_id).memory
        return memory

    def unlink_block_from_agent_memory(self, user_id: str, agent_id: str, block_label: str, delete_if_no_ref: bool = True) -> Memory:
        """Unlink a block from an agent's memory. If the block is not linked to any agent, delete it."""
        self.blocks_agents_manager.remove_block_with_label_from_agent(agent_id=agent_id, block_label=block_label)

        # get agent memory
        memory = self.get_agent(agent_id=agent_id).memory
        return memory

    def update_agent_memory_limit(self, user_id: str, agent_id: str, block_label: str, limit: int) -> Memory:
        """Update the limit of a block in an agent's memory"""
        block = self.get_agent_block_by_label(user_id=user_id, agent_id=agent_id, label=block_label)
        self.block_manager.update_block(
            block_id=block.id, block_update=BlockUpdate(limit=limit), actor=self.user_manager.get_user_by_id(user_id=user_id)
        )
        # get agent memory
        memory = self.get_agent(agent_id=agent_id).memory
        return memory

    def upate_block(self, user_id: str, block_id: str, block_update: BlockUpdate) -> Block:
        """Update a block"""
        return self.block_manager.update_block(
            block_id=block_id, block_update=block_update, actor=self.user_manager.get_user_by_id(user_id=user_id)
        )

    def get_agent_block_by_label(self, user_id: str, agent_id: str, label: str) -> Block:
        """Get a block by label"""
        # TODO: implement at ORM?
        for block_id in self.blocks_agents_manager.list_block_ids_for_agent(agent_id=agent_id):
            block = self.block_manager.get_block_by_id(block_id=block_id, actor=self.user_manager.get_user_by_id(user_id=user_id))
            if block.label == label:
                return block
        return None

    # def run_tool(self, tool_id: str, tool_args: str, user_id: str) -> FunctionReturn:
    #     """Run a tool using the sandbox and return the result"""

    #     try:
    #         tool_args_dict = json.loads(tool_args)
    #     except json.JSONDecodeError:
    #         raise ValueError("Invalid JSON string for tool_args")

    #     # Get the tool by ID
    #     user = self.user_manager.get_user_by_id(user_id=user_id)
    #     tool = self.tool_manager.get_tool_by_id(tool_id=tool_id, actor=user)
    #     if tool.name is None:
    #         raise ValueError(f"Tool with id {tool_id} does not have a name")

    #     # TODO eventually allow using agent state in tools
    #     agent_state = None

    #     try:
    #         sandbox_run_result = ToolExecutionSandbox(tool.name, tool_args_dict, user_id).run(agent_state=agent_state)
    #         if sandbox_run_result is None:
    #             raise ValueError(f"Tool with id {tool_id} returned execution with None")
    #         function_response = str(sandbox_run_result.func_return)

    #         return FunctionReturn(
    #             id="null",
    #             function_call_id="null",
    #             date=get_utc_time(),
    #             status="success",
    #             function_return=function_response,
    #         )
    #     except Exception as e:
    #         # same as agent.py
    #         from letta.constants import MAX_ERROR_MESSAGE_CHAR_LIMIT

    #         error_msg = f"Error executing tool {tool.name}: {e}"
    #         if len(error_msg) > MAX_ERROR_MESSAGE_CHAR_LIMIT:
    #             error_msg = error_msg[:MAX_ERROR_MESSAGE_CHAR_LIMIT]

    #         return FunctionReturn(
    #             id="null",
    #             function_call_id="null",
    #             date=get_utc_time(),
    #             status="error",
    #             function_return=error_msg,
    #         )

    def run_tool_from_source(
        self,
        user_id: str,
        tool_args: str,
        tool_source: str,
        tool_source_type: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> FunctionReturn:
        """Run a tool from source code"""

        try:
            tool_args_dict = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string for tool_args")

        if tool_source_type is not None and tool_source_type != "python":
            raise ValueError("Only Python source code is supported at this time")

        # NOTE: we're creating a floating Tool object and NOT persiting to DB
        tool = Tool(
            name=tool_name,
            source_code=tool_source,
        )
        assert tool.name is not None, "Failed to create tool object"

        # TODO eventually allow using agent state in tools
        agent_state = None

        # Next, attempt to run the tool with the sandbox
        try:
            sandbox_run_result = ToolExecutionSandbox(tool.name, tool_args_dict, user_id, tool_object=tool).run(agent_state=agent_state)
            if sandbox_run_result is None:
                raise ValueError(f"Tool with id {tool.id} returned execution with None")
            function_response = str(sandbox_run_result.func_return)

            return FunctionReturn(
                id="null",
                function_call_id="null",
                date=get_utc_time(),
                status="success",
                function_return=function_response,
            )
        except Exception as e:
            # same as agent.py
            from letta.constants import MAX_ERROR_MESSAGE_CHAR_LIMIT

            error_msg = f"Error executing tool {tool.name}: {e}"
            if len(error_msg) > MAX_ERROR_MESSAGE_CHAR_LIMIT:
                error_msg = error_msg[:MAX_ERROR_MESSAGE_CHAR_LIMIT]

            return FunctionReturn(
                id="null",
                function_call_id="null",
                date=get_utc_time(),
                status="error",
                function_return=error_msg,
            )

    # Composio wrappers
    def get_composio_client(self, api_key: Optional[str] = None):
        if api_key:
            return Composio(api_key=api_key)
        elif tool_settings.composio_api_key:
            return Composio(api_key=tool_settings.composio_api_key)
        else:
            return Composio()

    def get_composio_apps(self, api_key: Optional[str] = None) -> List["AppModel"]:
        """Get a list of all Composio apps with actions"""
        apps = self.get_composio_client(api_key=api_key).apps.get()
        apps_with_actions = []
        for app in apps:
            # A bit of hacky logic until composio patches this
            if app.meta["actionsCount"] > 0 and not app.name.lower().endswith("_beta"):
                apps_with_actions.append(app)

        return apps_with_actions

    def get_composio_actions_from_app_name(self, composio_app_name: str, api_key: Optional[str] = None) -> List["ActionModel"]:
        actions = self.get_composio_client(api_key=api_key).actions.get(apps=[composio_app_name])
        return actions
