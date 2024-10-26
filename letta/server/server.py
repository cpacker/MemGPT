# inspecting tools
import os
import traceback
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException

import letta.constants as constants
import letta.server.utils as server_utils
import letta.system as system
from letta.agent import Agent, save_agent
from letta.agent_store.db import attach_base
from letta.agent_store.storage import StorageConnector, TableType
from letta.credentials import LettaCredentials
from letta.data_sources.connectors import DataConnector, load_data

# from letta.data_types import (
#    AgentState,
#    EmbeddingConfig,
#    LLMConfig,
#    Message,
#    Preset,
#    Source,
#    Token,
#    User,
# )
from letta.functions.functions import generate_schema, parse_source_code
from letta.functions.schema_generator import generate_schema

# TODO use custom interface
from letta.interface import AgentInterface  # abstract
from letta.interface import CLIInterface  # for printing to terminal
from letta.log import get_logger
from letta.memory import get_memory_functions
from letta.metadata import Base, MetadataStore
from letta.o1_agent import O1Agent
from letta.prompts import gpt_system
from letta.providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
    LettaProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    VLLMChatCompletionsProvider,
    VLLMCompletionsProvider,
)
from letta.schemas.agent import AgentState, AgentType, CreateAgent, UpdateAgentState
from letta.schemas.api_key import APIKey, APIKeyCreate
from letta.schemas.block import (
    Block,
    CreateBlock,
    CreateHuman,
    CreatePersona,
    UpdateBlock,
)
from letta.schemas.embedding_config import EmbeddingConfig

# openai schemas
from letta.schemas.enums import JobStatus
from letta.schemas.file import FileMetadata
from letta.schemas.job import Job
from letta.schemas.letta_message import LettaMessage
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import (
    ArchivalMemorySummary,
    ContextWindowOverview,
    Memory,
    RecallMemorySummary,
)
from letta.schemas.message import Message, MessageCreate, MessageRole, UpdateMessage
from letta.schemas.passage import Passage
from letta.schemas.source import Source, SourceCreate, SourceUpdate
from letta.schemas.tool import Tool, ToolCreate
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager
from letta.utils import create_random_username, json_dumps, json_loads

# from letta.llm_api_tools import openai_get_model_list, azure_openai_get_model_list, smart_urljoin


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
        user_id: str,
        agent_config: Union[dict, AgentState],
        interface: Union[AgentInterface, None],
    ) -> str:
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


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

attach_base()

Base.metadata.create_all(bind=engine)


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

        # Make default user and org
        if init_with_default_org_and_user:
            self.default_org = self.organization_manager.create_default_organization()
            self.default_user = self.user_manager.create_default_user()
            self.add_default_blocks(self.default_user.id)
            self.tool_manager.add_default_tools(module_name="base", user_id=self.default_user.id, org_id=self.default_org.id)

            # If there is a default org/user
            # This logic may have to change in the future
            if settings.load_default_external_tools:
                self.add_default_external_tools(user_id=self.default_user.id, org_id=self.default_org.id)

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
                logger.debug(f"Saved agent {agent_d['agent_id']}")
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

    def _load_agent(self, user_id: str, agent_id: str, interface: Union[AgentInterface, None] = None) -> Agent:
        """Loads a saved agent into memory (if it doesn't exist, throw an error)"""
        assert isinstance(user_id, str), user_id
        assert isinstance(agent_id, str), agent_id

        # If an interface isn't specified, use the default
        if interface is None:
            interface = self.default_interface_factory()

        try:
            logger.debug(f"Grabbing agent user_id={user_id} agent_id={agent_id} from database")
            agent_state = self.ms.get_agent(agent_id=agent_id, user_id=user_id)
            if not agent_state:
                logger.exception(f"agent_id {agent_id} does not exist")
                raise ValueError(f"agent_id {agent_id} does not exist")

            # Instantiate an agent object using the state retrieved
            logger.debug(f"Creating an agent object")
            tool_objs = []
            for name in agent_state.tools:
                tool_obj = self.tool_manager.get_tool_by_name_and_user_id(tool_name=name, user_id=user_id)
                if not tool_obj:
                    logger.exception(f"Tool {name} does not exist for user {user_id}")
                    raise ValueError(f"Tool {name} does not exist for user {user_id}")
                tool_objs.append(tool_obj)

            # Make sure the memory is a memory object
            assert isinstance(agent_state.memory, Memory)

            if agent_state.agent_type == AgentType.memgpt_agent:
                letta_agent = Agent(agent_state=agent_state, interface=interface, tools=tool_objs)
            elif agent_state.agent_type == AgentType.o1_agent:
                letta_agent = O1Agent(agent_state=agent_state, interface=interface, tools=tool_objs)
            else:
                raise NotImplementedError("Not a supported agent type")

            # Add the agent to the in-memory store and return its reference
            logger.debug(f"Adding agent to the agent cache: user_id={user_id}, agent_id={agent_id}")
            self._add_agent(user_id=user_id, agent_id=agent_id, agent_obj=letta_agent)
            return letta_agent

        except Exception as e:
            logger.exception(f"Error occurred while trying to get agent {agent_id}:\n{e}")
            raise

    def _get_or_load_agent(self, agent_id: str) -> Agent:
        """Check if the agent is in-memory, then load"""
        agent_state = self.ms.get_agent(agent_id=agent_id)
        if not agent_state:
            raise ValueError(f"Agent does not exist")
        user_id = agent_state.user_id

        logger.debug(f"Checking for agent user_id={user_id} agent_id={agent_id}")
        # TODO: consider disabling loading cached agents due to potential concurrency issues
        letta_agent = self._get_agent(user_id=user_id, agent_id=agent_id)
        if not letta_agent:
            logger.debug(f"Agent not loaded, loading agent user_id={user_id} agent_id={agent_id}")
            letta_agent = self._load_agent(user_id=user_id, agent_id=agent_id)
        return letta_agent

    def _step(
        self,
        user_id: str,
        agent_id: str,
        input_messages: Union[Message, List[Message]],
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
            letta_agent = self._get_or_load_agent(agent_id=agent_id)
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
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
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
            ret_str = (
                f"\nDumping memory contents:\n"
                + f"\n{str(letta_agent.memory)}"
                + f"\n{str(letta_agent.persistence_manager.archival_memory)}"
                + f"\n{str(letta_agent.persistence_manager.recall_memory)}"
            )
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
                    user_id=user_id,
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    user_id=user_id,
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
                    user_id=user_id,
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    user_id=user_id,
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
                        user_id=user_id,
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
        return self._step(user_id=user_id, agent_id=agent_id, input_messages=message_objects)

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
        user_id: str,
        # interface
        interface: Union[AgentInterface, None] = None,
    ) -> AgentState:
        """Create a new agent using a config"""
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
            else:
                raise ValueError(f"Invalid agent type: {request.agent_type}")

        logger.debug(f"Attempting to find user: {user_id}")
        user = self.user_manager.get_user_by_id(user_id=user_id)
        if not user:
            raise ValueError(f"cannot find user with associated client id: {user_id}")

        try:
            # model configuration
            llm_config = request.llm_config
            embedding_config = request.embedding_config

            # get tools + make sure they exist
            tool_objs = []
            if request.tools:
                for tool_name in request.tools:
                    tool_obj = self.tool_manager.get_tool_by_name_and_user_id(tool_name=tool_name, user_id=user_id)
                    tool_objs.append(tool_obj)

            assert request.memory is not None
            memory_functions = get_memory_functions(request.memory)
            for func_name, func in memory_functions.items():

                if request.tools and func_name in request.tools:
                    # tool already added
                    continue
                source_code = parse_source_code(func)
                # memory functions are not terminal
                json_schema = generate_schema(func, terminal=False, name=func_name)
                source_type = "python"
                tags = ["memory", "memgpt-base"]
                tool = self.tool_manager.create_or_update_tool(
                    ToolCreate(
                        source_code=source_code,
                        source_type=source_type,
                        tags=tags,
                        json_schema=json_schema,
                        user_id=user_id,
                        organization_id=user.organization_id,
                    )
                )
                tool_objs.append(tool)
                if not request.tools:
                    request.tools = []
                request.tools.append(tool.name)

            # TODO: save the agent state
            agent_state = AgentState(
                name=request.name,
                user_id=user_id,
                tools=request.tools if request.tools else [],
                agent_type=request.agent_type or AgentType.memgpt_agent,
                llm_config=llm_config,
                embedding_config=embedding_config,
                system=request.system,
                memory=request.memory,
                description=request.description,
                metadata_=request.metadata_,
            )
            if request.agent_type == AgentType.memgpt_agent:
                agent = Agent(
                    interface=interface,
                    agent_state=agent_state,
                    tools=tool_objs,
                    # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
                    first_message_verify_mono=True if (llm_config.model is not None and "gpt-4" in llm_config.model) else False,
                )
            elif request.agent_type == AgentType.o1_agent:
                agent = O1Agent(
                    interface=interface,
                    agent_state=agent_state,
                    tools=tool_objs,
                    # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
                    first_message_verify_mono=True if (llm_config.model is not None and "gpt-4" in llm_config.model) else False,
                )
            # rebuilding agent memory on agent create in case shared memory blocks
            # were specified in the new agent's memory config. we're doing this for two reasons:
            # 1. if only the ID of the shared memory block was specified, we can fetch its most recent value
            # 2. if the shared block state changed since this agent initialization started, we can be sure to have the latest value
            agent.rebuild_memory(force=True, ms=self.ms)
            # FIXME: this is a hacky way to get the system prompts injected into agent into the DB
            # self.ms.update_agent(agent.agent_state)
        except Exception as e:
            logger.exception(e)
            try:
                if agent:
                    self.ms.delete_agent(agent_id=agent.agent_state.id)
            except Exception as delete_e:
                logger.exception(f"Failed to delete_agent:\n{delete_e}")
            raise e

        # save agent
        save_agent(agent, self.ms)
        logger.debug(f"Created new agent from config: {agent}")

        assert isinstance(agent.agent_state.memory, Memory), f"Invalid memory type: {type(agent_state.memory)}"
        # return AgentState
        return agent.agent_state

    def update_agent(
        self,
        request: UpdateAgentState,
        user_id: str,
    ):
        """Update the agents core memory block, return the new state"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=request.id) is None:
            raise ValueError(f"Agent agent_id={request.id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=request.id)

        # update the core memory of the agent
        if request.memory:
            assert isinstance(request.memory, Memory), type(request.memory)
            new_memory_contents = request.memory.to_flat_dict()
            _ = self.update_agent_core_memory(user_id=user_id, agent_id=request.id, new_memory_contents=new_memory_contents)

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
        if request.tools:
            # Replace tools and also re-link

            # (1) get tools + make sure they exist
            tool_objs = []
            for tool_name in request.tools:
                tool_obj = self.tool_manager.get_tool_by_name_and_user_id(tool_name=tool_name, user_id=user_id)
                assert tool_obj, f"Tool {tool_name} does not exist"
                tool_objs.append(tool_obj)

            # (2) replace the list of tool names ("ids") inside the agent state
            letta_agent.agent_state.tools = request.tools

            # (3) then attempt to link the tools modules
            letta_agent.link_tools(tool_objs)

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

        # save the agent
        assert isinstance(letta_agent.memory, Memory)
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
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.tools

    def add_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str,
        user_id: str,
    ):
        """Add tools from an existing agent"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # Get all the tool objects from the request
        tool_objs = []
        tool_obj = self.tool_manager.get_tool_by_id(tool_id=tool_id)
        assert tool_obj, f"Tool with id={tool_id} does not exist"
        tool_objs.append(tool_obj)

        for tool in letta_agent.tools:
            tool_obj = self.tool_manager.get_tool_by_id(tool_id=tool.id)
            assert tool_obj, f"Tool with id={tool.id} does not exist"

            # If it's not the already added tool
            if tool_obj.id != tool_id:
                tool_objs.append(tool_obj)

        # replace the list of tool names ("ids") inside the agent state
        letta_agent.agent_state.tools = [tool.name for tool in tool_objs]

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
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # Get all the tool_objs
        tool_objs = []
        for tool in letta_agent.tools:
            tool_obj = self.tool_manager.get_tool_by_id(tool_id=tool.id)
            assert tool_obj, f"Tool with id={tool.id} does not exist"

            # If it's not the tool we want to remove
            if tool_obj.id != tool_id:
                tool_objs.append(tool_obj)

        # replace the list of tool names ("ids") inside the agent state
        letta_agent.agent_state.tools = [tool.name for tool in tool_objs]

        # then attempt to link the tools modules
        letta_agent.link_tools(tool_objs)

        # save the agent
        save_agent(letta_agent, self.ms)
        return letta_agent.agent_state

    def _agent_state_to_config(self, agent_state: AgentState) -> dict:
        """Convert AgentState to a dict for a JSON response"""
        assert agent_state is not None

        agent_config = {
            "id": agent_state.id,
            "name": agent_state.name,
            "human": agent_state._metadata.get("human", None),
            "persona": agent_state._metadata.get("persona", None),
            "created_at": agent_state.created_at.isoformat(),
        }
        return agent_config

    def list_agents(
        self,
        user_id: str,
    ) -> List[AgentState]:
        """List all available agents to a user"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        agents_states = self.ms.list_agents(user_id=user_id)
        return agents_states

    def get_blocks(
        self,
        user_id: Optional[str] = None,
        label: Optional[str] = None,
        template: Optional[bool] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Optional[List[Block]]:

        return self.ms.get_blocks(user_id=user_id, label=label, template=template, name=name, id=id)

    def get_block(self, block_id: str):

        blocks = self.get_blocks(id=block_id)
        if blocks is None or len(blocks) == 0:
            raise ValueError("Block does not exist")
        if len(blocks) > 1:
            raise ValueError("Multiple blocks with the same id")
        return blocks[0]

    def create_block(self, request: CreateBlock, user_id: str, update: bool = False) -> Block:
        existing_blocks = self.ms.get_blocks(name=request.name, user_id=user_id, template=request.template, label=request.label)
        if existing_blocks is not None:
            existing_block = existing_blocks[0]
            assert len(existing_blocks) == 1
            if update:
                return self.update_block(UpdateBlock(id=existing_block.id, **vars(request)))
            else:
                raise ValueError(f"Block with name {request.name} already exists")
        block = Block(**vars(request))
        self.ms.create_block(block)
        return block

    def update_block(self, request: UpdateBlock) -> Block:
        block = self.get_block(request.id)
        block.limit = request.limit if request.limit is not None else block.limit
        block.value = request.value if request.value is not None else block.value
        block.name = request.name if request.name is not None else block.name
        self.ms.update_block(block=block)
        return self.ms.get_block(block_id=request.id)

    def delete_block(self, block_id: str):
        block = self.get_block(block_id)
        self.ms.delete_block(block_id)
        return block

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

    def get_agent(self, user_id: str, agent_id: Optional[str] = None, agent_name: Optional[str] = None):
        """Get the agent state"""
        return self.ms.get_agent(agent_id=agent_id, agent_name=agent_name, user_id=user_id)

    # def get_user(self, user_id: str) -> User:
    #     """Get the user"""
    #     user = self.user_manager.get_user_by_id(user_id=user_id)
    #     if user is None:
    #         raise ValueError(f"User with user_id {user_id} does not exist")
    #     else:
    #         return user

    def get_agent_memory(self, agent_id: str) -> Memory:
        """Return the memory of an agent (core memory)"""
        agent = self._get_or_load_agent(agent_id=agent_id)
        return agent.memory

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        agent = self._get_or_load_agent(agent_id=agent_id)
        return ArchivalMemorySummary(size=len(agent.persistence_manager.archival_memory))

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        agent = self._get_or_load_agent(agent_id=agent_id)
        return RecallMemorySummary(size=len(agent.persistence_manager.recall_memory))

    def get_in_context_message_ids(self, agent_id: str) -> List[str]:
        """Get the message ids of the in-context messages in the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return [m.id for m in letta_agent._messages]

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """Get the in-context messages in the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent._messages

    def get_agent_message(self, agent_id: str, message_id: str) -> Message:
        """Get a single message from the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        message = letta_agent.persistence_manager.recall_memory.storage.get(id=message_id)
        return message

    def get_agent_messages(
        self,
        agent_id: str,
        start: int,
        count: int,
        return_message_object: bool = True,
    ) -> Union[List[Message], List[LettaMessage]]:
        """Paginated query of all messages in agent message queue"""
        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

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

            ## Convert to json
            ## Add a tag indicating in-context or not
            # json_messages = [{**record.to_json(), "in_context": True} for record in messages]

        else:
            # need to access persistence manager for additional messages
            db_iterator = letta_agent.persistence_manager.recall_memory.storage.get_all_paginated(page_size=count, offset=start)

            # get a single page of messages
            # TODO: handle stop iteration
            page = next(db_iterator, [])

            # return messages in reverse chronological order
            messages = sorted(page, key=lambda x: x.created_at, reverse=True)
            assert all(isinstance(m, Message) for m in messages)

            ## Convert to json
            ## Add a tag indicating in-context or not
            # json_messages = [record.to_json() for record in messages]
            # in_context_message_ids = [str(m.id) for m in letta_agent._messages]
            # for d in json_messages:
            #    d["in_context"] = True if str(d["id"]) in in_context_message_ids else False

        if not return_message_object:
            messages = [msg for m in messages for msg in m.to_letta_message()]

        return messages

    def get_agent_archival(self, user_id: str, agent_id: str, start: int, count: int) -> List[Passage]:
        """Paginated query of all messages in agent archival memory"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # iterate over records
        db_iterator = letta_agent.persistence_manager.archival_memory.storage.get_all_paginated(page_size=count, offset=start)

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
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # iterate over recorde
        cursor, records = letta_agent.persistence_manager.archival_memory.storage.get_all_cursor(
            after=after, before=before, limit=limit, order_by=order_by, reverse=reverse
        )
        return records

    def insert_archival_memory(self, user_id: str, agent_id: str, memory_contents: str) -> List[Passage]:
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # Insert into archival memory
        passage_ids = letta_agent.persistence_manager.archival_memory.insert(memory_string=memory_contents, return_ids=True)

        # TODO: this is gross, fix
        return [letta_agent.persistence_manager.archival_memory.storage.get(id=passage_id) for passage_id in passage_ids]

    def delete_archival_memory(self, user_id: str, agent_id: str, memory_id: str):
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # TODO: should return a passage

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # Delete by ID
        # TODO check if it exists first, and throw error if not
        letta_agent.persistence_manager.archival_memory.storage.delete({"id": memory_id})

        # TODO: return archival memory

    def get_agent_recall_cursor(
        self,
        user_id: str,
        agent_id: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Optional[int] = 100,
        order_by: Optional[str] = "created_at",
        order: Optional[str] = "asc",
        reverse: Optional[bool] = False,
        return_message_object: bool = True,
        use_assistant_message: bool = False,
        assistant_message_function_name: str = constants.DEFAULT_MESSAGE_TOOL,
        assistant_message_function_kwarg: str = constants.DEFAULT_MESSAGE_TOOL_KWARG,
    ) -> Union[List[Message], List[LettaMessage]]:
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # iterate over records
        cursor, records = letta_agent.persistence_manager.recall_memory.storage.get_all_cursor(
            after=after, before=before, limit=limit, order_by=order_by, reverse=reverse
        )

        assert all(isinstance(m, Message) for m in records)

        if not return_message_object:
            # If we're GETing messages in reverse, we need to reverse the inner list (generated by to_letta_message)
            if reverse:
                records = [
                    msg
                    for m in records
                    for msg in m.to_letta_message(
                        assistant_message=use_assistant_message,
                        assistant_message_function_name=assistant_message_function_name,
                        assistant_message_function_kwarg=assistant_message_function_kwarg,
                    )[::-1]
                ]
            else:
                records = [
                    msg
                    for m in records
                    for msg in m.to_letta_message(
                        assistant_message=use_assistant_message,
                        assistant_message_function_name=assistant_message_function_name,
                        assistant_message_function_kwarg=assistant_message_function_kwarg,
                    )
                ]

        return records

    def get_agent_state(self, user_id: str, agent_id: Optional[str], agent_name: Optional[str] = None) -> Optional[AgentState]:
        """Return the config of an agent"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if agent_id:
            if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
                return None
        else:
            agent_state = self.ms.get_agent(agent_name=agent_name, user_id=user_id)
            if agent_state is None:
                raise ValueError(f"Agent agent_name={agent_name} does not exist")
            agent_id = agent_state.id

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        assert isinstance(letta_agent.memory, Memory)
        return letta_agent.agent_state.model_copy(deep=True)

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

    def update_agent_core_memory(self, user_id: str, agent_id: str, new_memory_contents: dict) -> Memory:
        """Update the agents core memory block, return the new state"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

        # old_core_memory = self.get_agent_memory(agent_id=agent_id)

        modified = False
        for key, value in new_memory_contents.items():
            if letta_agent.memory.get_block(key) is None:
                # raise ValueError(f"Key {key} not found in agent memory {list(letta_agent.memory.list_block_names())}")
                raise ValueError(f"Key {key} not found in agent memory {str(letta_agent.memory.memory)}")
            if value is None:
                continue
            if letta_agent.memory.get_block(key) != value:
                letta_agent.memory.update_block_value(label=key, value=value)  # update agent memory
                modified = True

        # If we modified the memory contents, we need to rebuild the memory block inside the system message
        if modified:
            letta_agent.rebuild_memory()
            # save agent
            save_agent(letta_agent, self.ms)

        return self.ms.get_agent(agent_id=agent_id).memory

    def rename_agent(self, user_id: str, agent_id: str, new_agent_name: str) -> AgentState:
        """Update the name of the agent in the database"""
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)

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
        if self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Verify that the agent exists and is owned by the user
        agent_state = self.ms.get_agent(agent_id=agent_id, user_id=user_id)
        if not agent_state:
            raise ValueError(f"Could not find agent_id={agent_id} under user_id={user_id}")
        if agent_state.user_id != user_id:
            raise ValueError(f"Could not authorize agent_id={agent_id} with user_id={user_id}")

        # First, if the agent is in the in-memory cache we should remove it
        # List of {'user_id': user_id, 'agent_id': agent_id, 'agent': agent_obj} dicts
        try:
            self.active_agents = [d for d in self.active_agents if str(d["agent_id"]) != str(agent_id)]
        except Exception as e:
            logger.exception(f"Failed to delete agent {agent_id} from cache via ID with:\n{str(e)}")
            raise ValueError(f"Failed to delete agent {agent_id} from cache")

        # Next, attempt to delete it from the actual database
        try:
            self.ms.delete_agent(agent_id=agent_id)
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

    def create_source(self, request: SourceCreate, user_id: str) -> Source:  # TODO: add other fields
        """Create a new data source"""
        source = Source(
            name=request.name,
            user_id=user_id,
            embedding_config=self.list_embedding_models()[0],  # TODO: require providing this
        )
        self.ms.create_source(source)
        assert self.ms.get_source(source_name=request.name, user_id=user_id) is not None, f"Failed to create source {request.name}"
        return source

    def update_source(self, request: SourceUpdate, user_id: str) -> Source:
        """Update an existing data source"""
        if not request.id:
            existing_source = self.ms.get_source(source_name=request.name, user_id=user_id)
        else:
            existing_source = self.ms.get_source(source_id=request.id)
        if not existing_source:
            raise ValueError("Source does not exist")

        # override updated fields
        if request.name:
            existing_source.name = request.name
        if request.metadata_:
            existing_source.metadata_ = request.metadata_
        if request.description:
            existing_source.description = request.description

        self.ms.update_source(existing_source)
        return existing_source

    def delete_source(self, source_id: str, user_id: str):
        """Delete a data source"""
        source = self.ms.get_source(source_id=source_id, user_id=user_id)
        self.ms.delete_source(source_id)

        # delete data from passage store
        passage_store = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)
        passage_store.delete({"source_id": source_id})

        # TODO: delete data from agent passage stores (?)

    def create_job(self, user_id: str, metadata: Optional[Dict] = None) -> Job:
        """Create a new job"""
        job = Job(
            user_id=user_id,
            status=JobStatus.created,
            metadata_=metadata,
        )
        self.ms.create_job(job)
        return job

    def delete_job(self, job_id: str):
        """Delete a job"""
        self.ms.delete_job(job_id)

    def get_job(self, job_id: str) -> Job:
        """Get a job"""
        return self.ms.get_job(job_id)

    def list_jobs(self, user_id: str) -> List[Job]:
        """List all jobs for a user"""
        return self.ms.list_jobs(user_id=user_id)

    def list_active_jobs(self, user_id: str) -> List[Job]:
        """List all active jobs for a user"""
        jobs = self.ms.list_jobs(user_id=user_id)
        return [job for job in jobs if job.status in [JobStatus.created, JobStatus.running]]

    def load_file_to_source(self, source_id: str, file_path: str, job_id: str) -> Job:

        # update job
        job = self.ms.get_job(job_id)
        job.status = JobStatus.running
        self.ms.update_job(job)

        # try:
        from letta.data_sources.connectors import DirectoryConnector

        source = self.ms.get_source(source_id=source_id)
        connector = DirectoryConnector(input_files=[file_path])
        num_passages, num_documents = self.load_data(user_id=source.user_id, source_name=source.name, connector=connector)
        # except Exception as e:
        #    # job failed with error
        #    error = str(e)
        #    print(error)
        #    job.status = JobStatus.failed
        #    job.metadata_["error"] = error
        #    self.ms.update_job(job)
        #    # TODO: delete any associated passages/files?

        #    # return failed job
        #    return job

        # update job status
        job.status = JobStatus.completed
        job.metadata_["num_passages"] = num_passages
        job.metadata_["num_documents"] = num_documents
        self.ms.update_job(job)

        return job

    def delete_file_from_source(self, source_id: str, file_id: str, user_id: Optional[str]) -> Optional[FileMetadata]:
        return self.ms.delete_file_from_source(source_id=source_id, file_id=file_id, user_id=user_id)

    def load_data(
        self,
        user_id: str,
        connector: DataConnector,
        source_name: str,
    ) -> Tuple[int, int]:
        """Load data from a DataConnector into a source for a specified user_id"""
        # TODO: this should be implemented as a batch job or at least async, since it may take a long time

        # load data from a data source into the document store
        source = self.ms.get_source(source_name=source_name, user_id=user_id)
        if source is None:
            raise ValueError(f"Data source {source_name} does not exist for user {user_id}")

        # get the data connectors
        passage_store = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)
        file_store = StorageConnector.get_storage_connector(TableType.FILES, self.config, user_id=user_id)

        # load data into the document store
        passage_count, document_count = load_data(connector, source, passage_store, file_store)
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
        data_source = self.ms.get_source(source_id=source_id, user_id=user_id, source_name=source_name)
        if data_source is None:
            raise ValueError(f"Data source id={source_id} name={source_name} does not exist for user_id {user_id}")

        # get connection to data source storage
        source_connector = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)

        # load agent
        agent = self._get_or_load_agent(agent_id=agent_id)

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
        if not source_id:
            assert source_name is not None, "source_name must be provided if source_id is not"
            source = self.ms.get_source(source_name=source_name, user_id=user_id)
            source_id = source.id
        else:
            source = self.ms.get_source(source_id=source_id)

        # delete all Passage objects with source_id==source_id from agent's archival memory
        agent = self._get_or_load_agent(agent_id=agent_id)
        archival_memory = agent.persistence_manager.archival_memory
        archival_memory.storage.delete({"source_id": source_id})

        # delete agent-source mapping
        self.ms.detach_source(agent_id=agent_id, source_id=source_id)

        # return back source data
        return source

    def list_attached_sources(self, agent_id: str) -> List[Source]:
        # list all attached sources to an agent
        return self.ms.list_attached_sources(agent_id)

    def list_files_from_source(self, source_id: str, limit: int = 1000, cursor: Optional[str] = None) -> List[FileMetadata]:
        # list all attached sources to an agent
        return self.ms.list_files_from_source(source_id=source_id, limit=limit, cursor=cursor)

    def list_data_source_passages(self, user_id: str, source_id: str) -> List[Passage]:
        warnings.warn("list_data_source_passages is not yet implemented, returning empty list.", category=UserWarning)
        return []

    def list_all_sources(self, user_id: str) -> List[Source]:
        """List all sources (w/ extra metadata) belonging to a user"""

        sources = self.ms.list_sources(user_id=user_id)

        # Add extra metadata to the sources
        sources_with_metadata = []
        for source in sources:

            # count number of passages
            passage_conn = StorageConnector.get_storage_connector(TableType.PASSAGES, self.config, user_id=user_id)
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
                    "name": self.ms.get_agent(user_id=user_id, agent_id=a_id).name,
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

    def add_default_external_tools(self, user_id: str, org_id: str) -> bool:
        """Add default langchain tools. Return true if successful, false otherwise."""
        success = True
        tool_creates = ToolCreate.load_default_langchain_tools() + ToolCreate.load_default_crewai_tools()
        if tool_settings.composio_api_key:
            tool_creates += ToolCreate.load_default_composio_tools()
        for tool_create in tool_creates:
            try:
                self.tool_manager.create_or_update_tool(tool_create)
            except Exception as e:
                warnings.warn(f"An error occurred while creating tool {tool_create}: {e}")
                warnings.warn(traceback.format_exc())
                success = False

        return success

    def add_default_blocks(self, user_id: str):
        from letta.utils import list_human_files, list_persona_files

        assert user_id is not None, "User ID must be provided"

        for persona_file in list_persona_files():
            text = open(persona_file, "r", encoding="utf-8").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            self.create_block(CreatePersona(user_id=user_id, name=name, value=text, template=True), user_id=user_id, update=True)

        for human_file in list_human_files():
            text = open(human_file, "r", encoding="utf-8").read()
            name = os.path.basename(human_file).replace(".txt", "")
            self.create_block(CreateHuman(user_id=user_id, name=name, value=text, template=True), user_id=user_id, update=True)

    def get_agent_message(self, agent_id: str, message_id: str) -> Optional[Message]:
        """Get a single message from the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        message = letta_agent.persistence_manager.recall_memory.storage.get(id=message_id)
        return message

    def update_agent_message(self, agent_id: str, request: UpdateMessage) -> Message:
        """Update the details of a message associated with an agent"""

        # Get the current message
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.update_message(request=request)

        # TODO decide whether this should be done in the server.py or agent.py
        # Reason to put it in agent.py:
        # - we use the agent object's persistence_manager to update the message
        # - it makes it easy to do things like `retry`, `rethink`, etc.
        # Reason to put it in server.py:
        # - fundamentally, we should be able to edit a message (without agent id)
        #   in the server by directly accessing the DB / message store
        """
        message = letta_agent.persistence_manager.recall_memory.storage.get(id=request.id)
        if message is None:
            raise ValueError(f"Message with id {request.id} not found")

        # Override fields
        # NOTE: we try to do some sanity checking here (see asserts), but it's not foolproof
        if request.role:
            message.role = request.role
        if request.text:
            message.text = request.text
        if request.name:
            message.name = request.name
        if request.tool_calls:
            assert message.role == MessageRole.assistant, "Tool calls can only be added to assistant messages"
            message.tool_calls = request.tool_calls
        if request.tool_call_id:
            assert message.role == MessageRole.tool, "tool_call_id can only be added to tool messages"
            message.tool_call_id = request.tool_call_id

        # Save the updated message
        letta_agent.persistence_manager.recall_memory.storage.update(record=message)

        # Return the updated message
        updated_message = letta_agent.persistence_manager.recall_memory.storage.get(id=message.id)
        if updated_message is None:
            raise ValueError(f"Error persisting message - message with id {request.id} not found")
        return updated_message
        """

    def rewrite_agent_message(self, agent_id: str, new_text: str) -> Message:

        # Get the current message
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.rewrite_message(new_text=new_text)

    def rethink_agent_message(self, agent_id: str, new_thought: str) -> Message:

        # Get the current message
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.rethink_message(new_thought=new_thought)

    def retry_agent_message(self, agent_id: str) -> List[Message]:

        # Get the current message
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.retry_message()

    def get_user_or_default(self, user_id: Optional[str]) -> User:
        """Get the user object for user_id if it exists, otherwise return the default user object"""
        if user_id is None:
            user_id = self.user_manager.DEFAULT_USER_ID

        try:
            return self.user_manager.get_user_by_id(user_id=user_id)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"User with id {user_id} not found")

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
        letta_agent = self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.get_context_window()
