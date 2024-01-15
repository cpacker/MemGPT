from abc import abstractmethod
from typing import Union, Callable
import uuid
import json
import logging
from threading import Lock
from functools import wraps
from fastapi import HTTPException

from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.agent import Agent
import memgpt.system as system
import memgpt.constants as constants
from memgpt.cli.cli import attach

# from memgpt.agent_store.storage import StorageConnector
from memgpt.metadata import MetadataStore
import memgpt.presets.presets as presets
import memgpt.utils as utils
import memgpt.server.utils as server_utils
from memgpt.persistence_manager import PersistenceManager, LocalStateManager
from memgpt.data_types import Source, Passage, Document, User, AgentState

# TODO use custom interface
from memgpt.interface import CLIInterface  # for printing to terminal
from memgpt.interface import AgentInterface  # abstract

logger = logging.getLogger(__name__)


class Server(object):
    """Abstract server class that supports multi-agent multi-user"""

    @abstractmethod
    def list_agents(self, user_id: uuid.UUID) -> dict:
        """List all available agents to a user"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_messages(self, user_id: uuid.UUID, agent_id: uuid.UUID, start: int, count: int) -> list:
        """Paginated query of in-context messages in agent message queue"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_memory(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_config(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> dict:
        """Return the config of an agent"""
        raise NotImplementedError

    @abstractmethod
    def get_server_config(self, user_id: uuid.UUID) -> dict:
        """Return the base config"""
        raise NotImplementedError

    @abstractmethod
    def update_agent_core_memory(self, user_id: uuid.UUID, agent_id: uuid.UUID, new_memory_contents: dict) -> dict:
        """Update the agents core memory block, return the new state"""
        raise NotImplementedError

    @abstractmethod
    def create_agent(
        self,
        user_id: uuid.UUID,
        agent_config: Union[dict, AgentState],
        interface: Union[AgentInterface, None],
        # persistence_manager: Union[PersistenceManager, None],
    ) -> str:
        """Create a new agent using a config"""
        raise NotImplementedError

    @abstractmethod
    def user_message(self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str) -> None:
        """Process a message from the user, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def system_message(self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str) -> None:
        """Process a message from the system, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def run_command(self, user_id: uuid.UUID, agent_id: uuid.UUID, command: str) -> Union[str, None]:
        """Run a command on the agent, e.g. /memory

        May return a string with a message generated by the command
        """
        raise NotImplementedError


class LockingServer(Server):
    """Basic support for concurrency protections (all requests that modify an agent lock the agent until the operation is complete)"""

    # Locks for each agent
    _agent_locks = {}

    @staticmethod
    def agent_lock_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, user_id: uuid.UUID, agent_id: uuid.UUID, *args, **kwargs):
            # logger.info("Locking check")

            # Initialize the lock for the agent_id if it doesn't exist
            if agent_id not in self._agent_locks:
                # logger.info(f"Creating lock for agent_id = {agent_id}")
                self._agent_locks[agent_id] = Lock()

            # Check if the agent is currently locked
            if not self._agent_locks[agent_id].acquire(blocking=False):
                # logger.info(f"agent_id = {agent_id} is busy")
                raise HTTPException(status_code=423, detail=f"Agent '{agent_id}' is currently busy.")

            try:
                # Execute the function
                # logger.info(f"running function on agent_id = {agent_id}")
                return func(self, user_id, agent_id, *args, **kwargs)
            finally:
                # Release the lock
                # logger.info(f"releasing lock on agent_id = {agent_id}")
                self._agent_locks[agent_id].release()

        return wrapper

    @agent_lock_decorator
    def user_message(self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str) -> None:
        raise NotImplementedError

    @agent_lock_decorator
    def run_command(self, user_id: uuid.UUID, agent_id: uuid.UUID, command: str) -> Union[str, None]:
        raise NotImplementedError


# TODO actually use "user_id" for something
class SyncServer(LockingServer):
    """Simple single-threaded / blocking server process"""

    def __init__(
        self,
        chaining: bool = True,
        max_chaining_steps: bool = None,
        # default_interface_cls: AgentInterface = CLIInterface,
        default_interface: AgentInterface = CLIInterface(),
        # default_persistence_manager_cls: PersistenceManager = LocalStateManager,
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
        # self.default_interface_cls = default_interface_cls
        self.default_interface = default_interface

        # The default persistence manager that will get assigned to agents ON CREATION
        # self.default_persistence_manager_cls = default_persistence_manager_cls

        # Initialize the connection to the DB
        self.config = MemGPTConfig.load()
        self.ms = MetadataStore(self.config)

        # Create the default user
        base_user_id = uuid.UUID(self.config.anon_clientid)
        if not self.ms.get_user(user_id=base_user_id):
            base_user = User(id=base_user_id)
            self.ms.create_user(base_user)

    def save_agents(self):
        """Saves all the agents that are in the in-memory object store"""
        for agent_d in self.active_agents:
            try:
                agent_d["agent"].save()
                logger.info(f"Saved agent {agent_d['agent_id']}")
            except Exception as e:
                logger.exception(f"Error occurred while trying to save agent {agent_d['agent_id']}:\n{e}")

    def _get_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> Union[Agent, None]:
        """Get the agent object from the in-memory object store"""
        for d in self.active_agents:
            if d["user_id"] == str(user_id) and d["agent_id"] == str(agent_id):
                return d["agent"]
        return None

    def _add_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID, agent_obj: Agent) -> None:
        """Put an agent object inside the in-memory object store"""
        # Make sure the agent doesn't already exist
        if self._get_agent(user_id=user_id, agent_id=agent_id) is not None:
            raise KeyError(f"Agent (user={user_id}, agent={agent_id}) is already loaded")
        # Add Agent instance to the in-memory list
        self.active_agents.append(
            {
                "user_id": str(user_id),
                "agent_id": str(agent_id),
                "agent": agent_obj,
            }
        )

    def _load_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID, interface: Union[AgentInterface, None] = None) -> Agent:
        """Loads a saved agent into memory (if it doesn't exist, throw an error)"""
        assert isinstance(user_id, uuid.UUID), user_id
        assert isinstance(agent_id, uuid.UUID), agent_id

        # If an interface isn't specified, use the default
        if interface is None:
            interface = self.default_interface

        try:
            logger.info(f"Grabbing agent user_id={user_id} agent_id={agent_id} from database")
            agent_state = self.ms.get_agent(agent_id=agent_id, user_id=user_id)
            if not agent_state:
                logger.exception(f"agent_id {agent_id} does not exist")
                raise ValueError(f"agent_id {agent_id} does not exist")

            # Instantiate an agent object using the state retrieved
            logger.info(f"Creating an agent object")
            memgpt_agent = Agent(agent_state=agent_state, interface=interface)

            # Add the agent to the in-memory store and return its reference
            self._add_agent(user_id=user_id, agent_id=agent_id, agent_obj=memgpt_agent)
            logger.info(f"Creating an agent object")
            return memgpt_agent

        except Exception as e:
            logger.exception(f"Error occurred while trying to get agent {agent_id}:\n{e}")

    def _get_or_load_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> Agent:
        """Check if the agent is in-memory, then load"""
        memgpt_agent = self._get_agent(user_id=user_id, agent_id=agent_id)
        if not memgpt_agent:
            logger.info(f"Loading agent user_id={user_id} agent_id={agent_id}")
            memgpt_agent = self._load_agent(user_id=user_id, agent_id=agent_id)
        return memgpt_agent

    def _step(self, user_id: uuid.UUID, agent_id: uuid.UUID, input_message: str) -> None:
        """Send the input message through the agent"""

        logger.debug(f"Got input message: {input_message}")

        # Get the agent object (loaded in memory)
        memgpt_agent = self._get_or_load_agent(user_id=user_id, agent_id=agent_id)
        if memgpt_agent is None:
            raise KeyError(f"Agent (user={user_id}, agent={agent_id}) is not loaded")

        logger.debug(f"Starting agent step")
        no_verify = True
        next_input_message = input_message
        counter = 0
        while True:
            new_messages, heartbeat_request, function_failed, token_warning = memgpt_agent.step(
                next_input_message, first_message=False, skip_verify=no_verify
            )
            counter += 1

            # Chain stops
            if not self.chaining:
                logger.debug("No chaining, stopping after one step")
                break
            elif self.max_chaining_steps is not None and counter > self.max_chaining_steps:
                logger.debug(f"Hit max chaining steps, stopping after {counter} steps")
                break
            # Chain handlers
            elif token_warning:
                next_input_message = system.get_token_limit_warning()
                continue  # always chain
            elif function_failed:
                next_input_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                continue  # always chain
            elif heartbeat_request:
                next_input_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                continue  # always chain
            # MemGPT no-op / yield
            else:
                break

        memgpt_agent.interface.step_yield()
        logger.debug(f"Finished agent step")

    def _command(self, user_id: uuid.UUID, agent_id: uuid.UUID, command: str) -> Union[str, None]:
        """Process a CLI command"""

        logger.debug(f"Got command: {command}")

        # Get the agent object (loaded in memory)
        memgpt_agent = self._get_or_load_agent(user_id=user_id, agent_id=agent_id)

        if command.lower() == "exit":
            # exit not supported on server.py
            raise ValueError(command)

        elif command.lower() == "save" or command.lower() == "savechat":
            memgpt_agent.save()

        elif command.lower() == "attach":
            # Different from CLI, we extract the data source name from the command
            command = command.strip().split()
            try:
                data_source = int(command[1])
            except:
                raise ValueError(command)

            # TODO: check if agent already has it
            data_source_options = StorageConnector.list_loaded_data()
            if len(data_source_options) == 0:
                raise ValueError('No sources available. You must load a souce with "memgpt load ..." before running /attach.')
            elif data_source not in data_source_options:
                raise ValueError(f"Invalid data source name: {data_source} (options={data_source_options})")
            else:
                # attach new data
                attach(memgpt_agent.config.name, data_source)

                # update agent config
                memgpt_agent.config.attach_data_source(data_source)

                # reload agent with new data source
                # TODO: maybe make this less ugly...
                memgpt_agent.persistence_manager.archival_memory.storage = StorageConnector.get_storage_connector(
                    agent_config=memgpt_agent.config
                )

        elif command.lower() == "dump" or command.lower().startswith("dump "):
            # Check if there's an additional argument that's an integer
            command = command.strip().split()
            amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 0
            if amount == 0:
                memgpt_agent.interface.print_messages(memgpt_agent.messages, dump=True)
            else:
                memgpt_agent.interface.print_messages(memgpt_agent.messages[-min(amount, len(memgpt_agent.messages)) :], dump=True)

        elif command.lower() == "dumpraw":
            memgpt_agent.interface.print_messages_raw(memgpt_agent.messages)

        elif command.lower() == "memory":
            ret_str = (
                f"\nDumping memory contents:\n"
                + f"\n{str(memgpt_agent.memory)}"
                + f"\n{str(memgpt_agent.persistence_manager.archival_memory)}"
                + f"\n{str(memgpt_agent.persistence_manager.recall_memory)}"
            )
            return ret_str

        elif command.lower() == "pop" or command.lower().startswith("pop "):
            # Check if there's an additional argument that's an integer
            command = command.strip().split()
            pop_amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
            n_messages = len(memgpt_agent.messages)
            MIN_MESSAGES = 2
            if n_messages <= MIN_MESSAGES:
                logger.info(f"Agent only has {n_messages} messages in stack, none left to pop")
            elif n_messages - pop_amount < MIN_MESSAGES:
                logger.info(f"Agent only has {n_messages} messages in stack, cannot pop more than {n_messages - MIN_MESSAGES}")
            else:
                logger.info(f"Popping last {pop_amount} messages from stack")
                for _ in range(min(pop_amount, len(memgpt_agent.messages))):
                    memgpt_agent.messages.pop()

        elif command.lower() == "retry":
            # TODO this needs to also modify the persistence manager
            logger.info(f"Retrying for another answer")
            while len(memgpt_agent.messages) > 0:
                if memgpt_agent.messages[-1].get("role") == "user":
                    # we want to pop up to the last user message and send it again
                    user_message = memgpt_agent.messages[-1].get("content")
                    memgpt_agent.messages.pop()
                    break
                memgpt_agent.messages.pop()

        elif command.lower() == "rethink" or command.lower().startswith("rethink "):
            # TODO this needs to also modify the persistence manager
            if len(command) < len("rethink "):
                logger.warning("Missing text after the command")
            else:
                for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                    if memgpt_agent.messages[x].get("role") == "assistant":
                        text = command[len("rethink ") :].strip()
                        memgpt_agent.messages[x].update({"content": text})
                        break

        elif command.lower() == "rewrite" or command.lower().startswith("rewrite "):
            # TODO this needs to also modify the persistence manager
            if len(command) < len("rewrite "):
                logger.warning("Missing text after the command")
            else:
                for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                    if memgpt_agent.messages[x].get("role") == "assistant":
                        text = command[len("rewrite ") :].strip()
                        args = json.loads(memgpt_agent.messages[x].get("function_call").get("arguments"))
                        args["message"] = text
                        memgpt_agent.messages[x].get("function_call").update(
                            {"arguments": json.dumps(args, ensure_ascii=constants.JSON_ENSURE_ASCII)}
                        )
                        break

        # No skip options
        elif command.lower() == "wipe":
            # exit not supported on server.py
            raise ValueError(command)

        elif command.lower() == "heartbeat":
            input_message = system.get_heartbeat()
            self._step(user_id=user_id, agent_id=agent_id, input_message=input_message)

        elif command.lower() == "memorywarning":
            input_message = system.get_token_limit_warning()
            self._step(user_id=user_id, agent_id=agent_id, input_message=input_message)

    @LockingServer.agent_lock_decorator
    def user_message(self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str) -> None:
        """Process an incoming user message and feed it through the MemGPT agent"""

        # Basic input sanitization
        if not isinstance(message, str) or len(message) == 0:
            raise ValueError(f"Invalid input: '{message}'")

        # If the input begins with a command prefix, reject
        elif message.startswith("/"):
            raise ValueError(f"Invalid input: '{message}'")

        # Else, process it as a user message to be fed to the agent
        else:
            # Package the user message first
            packaged_user_message = system.package_user_message(user_message=message)
            # Run the agent state forward
            self._step(user_id=user_id, agent_id=agent_id, input_message=packaged_user_message)

    @LockingServer.agent_lock_decorator
    def system_message(self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str) -> None:
        """Process an incoming system message and feed it through the MemGPT agent"""
        from memgpt.utils import printd

        # Basic input sanitization
        if not isinstance(message, str) or len(message) == 0:
            raise ValueError(f"Invalid input: '{message}'")

        # If the input begins with a command prefix, reject
        elif message.startswith("/"):
            raise ValueError(f"Invalid input: '{message}'")

        # Else, process it as a user message to be fed to the agent
        else:
            # Package the user message first
            packaged_system_message = system.package_system_message(system_message=message)
            # Run the agent state forward
            self._step(user_id=user_id, agent_id=agent_id, input_message=packaged_system_message)

    @LockingServer.agent_lock_decorator
    def run_command(self, user_id: uuid.UUID, agent_id: uuid.UUID, command: str) -> Union[str, None]:
        """Run a command on the agent"""
        # If the input begins with a command prefix, attempt to process it as a command
        if command.startswith("/"):
            if len(command) > 1:
                command = command[1:]  # strip the prefix
        return self._command(user_id=user_id, agent_id=agent_id, command=command)

    def create_agent(
        self,
        user_id: uuid.UUID,
        agent_config: Union[dict, AgentState],
        interface: Union[AgentInterface, None] = None,
        # persistence_manager: Union[PersistenceManager, None] = None,
    ) -> AgentState:
        """Create a new agent using a config"""

        # Initialize the agent based on the provided configuration
        if not isinstance(agent_config, dict):
            raise ValueError(f"agent_config must be provided as a dictionary")

        if interface is None:
            # interface = self.default_interface_cls()
            interface = self.default_interface

        # if persistence_manager is None:
        # persistence_manager = self.default_persistence_manager_cls(agent_config=agent_config)

        # TODO actually use the user_id that was passed into the server
        user_id = uuid.UUID(self.config.anon_clientid)

        logger.debug(f"Attempting to find user: {user_id}")
        user = self.ms.get_user(user_id=user_id)
        if not user:
            raise ValueError(f"cannot find user with associated client id: {user_id}")

        agent_state = AgentState(
            user_id=user.id,
            name=agent_config["name"] if "name" in agent_config else utils.create_random_username(),
            preset=agent_config["preset"] if "preset" in agent_config else user.default_preset,
            # TODO we need to allow passing raw persona/human text via the server request
            persona=agent_config["persona"] if "persona" in agent_config else user.default_persona,
            human=agent_config["human"] if "human" in agent_config else user.default_human,
            llm_config=agent_config["llm_config"] if "llm_config" in agent_config else user.default_llm_config,
            embedding_config=agent_config["embedding_config"] if "embedding_config" in agent_config else user.default_embedding_config,
        )
        logger.debug(f"Attempting to create agent from agent_state:\n{agent_state}")
        try:
            agent = presets.create_agent_from_preset(agent_state=agent_state, interface=interface)
        except Exception as e:
            logger.exception(e)
            raise

        logger.info(f"Created new agent from config: {agent}")

        return agent.config

    def delete_agent(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
    ):
        # Make sure the user owns the agent
        # TODO use real user_id
        USER_ID = self.config.anon_clientid
        agent = self.ms.get_agent(agent_id=agent_id, user_id=USER_ID)
        if agent is not None:
            self.ms.delete_agent(agent_id=agent_id)

    def list_agents(self, user_id: uuid.UUID) -> dict:
        """List all available agents to a user"""
        # TODO actually use the user_id that was passed into the server
        user_id = uuid.UUID(self.config.anon_clientid)
        agents_states = self.ms.list_agents(user_id=user_id)
        logger.info(f"Retrieved {len(agents_states)} agents for user {user_id}:\n{[vars(s) for s in agents_states]}")
        return {
            "num_agents": len(agents_states),
            "agents": [
                {
                    "id": state.id,
                    "name": state.name,
                    "human": state.human,
                    "persona": state.persona,
                    "created_at": state.created_at.isoformat(),
                }
                for state in agents_states
            ],
        }

    def get_agent_memory(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        # Get the agent object (loaded in memory)
        # TODO: use real user_id
        memgpt_agent = self._get_or_load_agent(user_id=self.config.anon_clientid, agent_id=agent_id)

        core_memory = memgpt_agent.memory
        recall_memory = memgpt_agent.persistence_manager.recall_memory
        archival_memory = memgpt_agent.persistence_manager.archival_memory

        memory_obj = {
            "core_memory": {
                "persona": core_memory.persona,
                "human": core_memory.human,
            },
            "recall_memory": len(recall_memory) if recall_memory is not None else None,
            "archival_memory": len(archival_memory) if archival_memory is not None else None,
        }

        return memory_obj

    def get_agent_messages(self, user_id: uuid.UUID, agent_id: uuid.UUID, start: int, count: int) -> list:
        """Paginated query of in-context messages in agent message queue"""
        # Get the agent object (loaded in memory)
        memgpt_agent = self._get_or_load_agent(user_id=user_id, agent_id=agent_id)

        if start < 0 or count < 0:
            raise ValueError("Start and count values should be non-negative")

        if start + count < len(memgpt_agent.messages):  # messages can be returned from whats in memory
            # Reverse the list to make it in reverse chronological order
            reversed_messages = memgpt_agent.messages[::-1]
            # Check if start is within the range of the list
            if start >= len(reversed_messages):
                raise IndexError("Start index is out of range")

            # Calculate the end index, ensuring it does not exceed the list length
            end_index = min(start + count, len(reversed_messages))

            # Slice the list for pagination
            paginated_messages = reversed_messages[start:end_index]
            return paginated_messages

        # need to access persistence manager for additional messages
        filters = {"agent_id": agent_id, "user_id": user_id}
        db_iterator = memgpt_agent.persistence_manager.recall_memory.storage.get_all_paginated(filters=filters, limit=count, offset=start)

        # get a single page of messages
        messages = next(db_iterator)

        # return messages in reverse chronological order
        revered_messages = sorted(messages, key=lambda x: x.created_at, reverse=True)
        return [vars(m) for m in revered_messages]

    def get_agent_config(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> dict:
        """Return the config of an agent"""
        # Get the agent object (loaded in memory)
        memgpt_agent = self._get_or_load_agent(user_id=user_id, agent_id=agent_id)
        agent_config = vars(memgpt_agent.config)

        return agent_config

    def get_server_config(self) -> dict:
        """Return the base config"""
        # TODO: do we need a seperate server config?
        base_config = vars(self.config)

        def clean_keys(config):
            config_copy = config.copy()
            for k, v in config.items():
                if k == "key" or "_key" in k:
                    config_copy[k] = server_utils.shorten_key_middle(v, chars_each_side=5)
            return config_copy

        clean_base_config = clean_keys(base_config)
        return clean_base_config

    def update_agent_core_memory(self, user_id: uuid.UUID, agent_id: uuid.UUID, new_memory_contents: dict) -> dict:
        """Update the agents core memory block, return the new state"""
        # Get the agent object (loaded in memory)
        memgpt_agent = self._get_or_load_agent(user_id=user_id, agent_id=agent_id)

        old_core_memory = self.get_agent_memory(user_id=user_id, agent_id=agent_id)["core_memory"]
        new_core_memory = old_core_memory.copy()

        modified = False
        if "persona" in new_memory_contents and new_memory_contents["persona"] is not None:
            new_persona = new_memory_contents["persona"]
            if old_core_memory["persona"] != new_persona:
                new_core_memory["persona"] = new_persona
                memgpt_agent.memory.edit_persona(new_persona)
                modified = True

        if "human" in new_memory_contents and new_memory_contents["human"] is not None:
            new_human = new_memory_contents["human"]
            if old_core_memory["human"] != new_human:
                new_core_memory["human"] = new_human
                memgpt_agent.memory.edit_human(new_human)
                modified = True

        # If we modified the memory contents, we need to rebuild the memory block inside the system message
        if modified:
            memgpt_agent.rebuild_memory()

        return {
            "old_core_memory": old_core_memory,
            "new_core_memory": new_core_memory,
            "modified": modified,
        }
