import uuid
from typing import Callable, Optional, List, Dict, Union, Any, Tuple

from autogen.agentchat import Agent, ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager

from memgpt.agent import Agent as MemGPTAgent
from memgpt.autogen.interface import AutoGenInterface
import memgpt.system as system
import memgpt.constants as constants
import memgpt.utils as utils
import memgpt.presets.presets as presets
from memgpt.config import MemGPTConfig
from memgpt.credentials import MemGPTCredentials
from memgpt.cli.cli import attach
from memgpt.cli.cli_load import load_directory, load_webpage, load_index, load_database, load_vector_database
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.metadata import MetadataStore, save_agent
from memgpt.data_types import AgentState, User, LLMConfig, EmbeddingConfig


def create_memgpt_autogen_agent_from_config(
    name: str,
    system_message: Optional[str] = "You are a helpful AI Assistant.",
    is_termination_msg: Optional[Callable[[Dict], bool]] = None,
    max_consecutive_auto_reply: Optional[int] = None,
    human_input_mode: Optional[str] = "ALWAYS",
    function_map: Optional[Dict[str, Callable]] = None,
    code_execution_config: Optional[Union[Dict, bool]] = None,
    llm_config: Optional[Union[Dict, bool]] = None,
    # config setup for non-memgpt agents:
    nonmemgpt_llm_config: Optional[Union[Dict, bool]] = None,
    default_auto_reply: Optional[Union[str, Dict, None]] = "",
    interface_kwargs: Dict = None,
    skip_verify: bool = False,
):
    """Same function signature as used in base AutoGen, but creates a MemGPT agent

    Construct AutoGen config workflow in a clean way.
    """
    llm_config = llm_config["config_list"][0]

    if interface_kwargs is None:
        interface_kwargs = {}

    # The "system message" in AutoGen becomes the persona in MemGPT
    persona_desc = utils.get_persona_text(constants.DEFAULT_PERSONA) if system_message == "" else system_message
    # The user profile is based on the input mode
    if human_input_mode == "ALWAYS":
        user_desc = ""
    elif human_input_mode == "TERMINATE":
        user_desc = "Work by yourself, the user won't reply until you output `TERMINATE` to end the conversation."
    else:
        user_desc = "Work by yourself, the user won't reply. Elaborate as much as possible."

    # If using azure or openai, save the credentials to the config
    config = MemGPTConfig.load()
    credentials = MemGPTCredentials.load()

    if (
        llm_config["model_endpoint_type"] in ["azure", "openai"]
        or llm_config["model_endpoint_type"] != config.default_llm_config.model_endpoint_type
    ):
        # we load here to make sure we don't override existing values
        # all we want to do is add extra credentials

        if llm_config["model_endpoint_type"] == "azure":
            credentials.azure_key = llm_config["azure_key"]
            credentials.azure_endpoint = llm_config["azure_endpoint"]
            credentials.azure_version = llm_config["azure_version"]
            llm_config.pop("azure_key")
            llm_config.pop("azure_endpoint")
            llm_config.pop("azure_version")

        elif llm_config["model_endpoint_type"] == "openai":
            credentials.openai_key = llm_config["openai_key"]
            llm_config.pop("openai_key")

        credentials.save()

    # Create an AgentConfig option from the inputs
    llm_config.pop("name", None)
    llm_config.pop("persona", None)
    llm_config.pop("human", None)
    agent_config = dict(
        name=name,
        persona=persona_desc,
        human=user_desc,
        **llm_config,
    )

    if function_map is not None or code_execution_config is not None:
        raise NotImplementedError

    autogen_memgpt_agent = create_autogen_memgpt_agent(
        agent_config,
        default_auto_reply=default_auto_reply,
        is_termination_msg=is_termination_msg,
        interface_kwargs=interface_kwargs,
        skip_verify=skip_verify,
    )

    if human_input_mode != "ALWAYS":
        coop_agent1 = create_autogen_memgpt_agent(
            agent_config,
            default_auto_reply=default_auto_reply,
            is_termination_msg=is_termination_msg,
            interface_kwargs=interface_kwargs,
            skip_verify=skip_verify,
        )
        if default_auto_reply != "":
            coop_agent2 = UserProxyAgent(
                "User_proxy",
                human_input_mode="NEVER",
                default_auto_reply=default_auto_reply,
            )
        else:
            coop_agent2 = create_autogen_memgpt_agent(
                agent_config,
                default_auto_reply=default_auto_reply,
                is_termination_msg=is_termination_msg,
                interface_kwargs=interface_kwargs,
                skip_verify=skip_verify,
            )

        groupchat = GroupChat(
            agents=[autogen_memgpt_agent, coop_agent1, coop_agent2],
            messages=[],
            max_round=12 if max_consecutive_auto_reply is None else max_consecutive_auto_reply,
        )
        assert nonmemgpt_llm_config is not None
        manager = GroupChatManager(name=name, groupchat=groupchat, llm_config=nonmemgpt_llm_config)
        return manager

    else:
        return autogen_memgpt_agent


def update_config_from_dict(config_object: Union[LLMConfig, EmbeddingConfig], config_dict: dict):
    """
    Update the attributes of a configuration object based on a dictionary.

    :param config_object: The configuration object to be updated.
    :param config_dict: The dictionary containing new values for the configuration.
    """
    for attr in dir(config_object):
        # Filter out private attributes and methods
        if not attr.startswith("_") and not callable(getattr(config_object, attr)):
            if attr in config_dict:
                # Cast the value to the type of the attribute in config_object
                attr_type = type(getattr(config_object, attr))
                try:
                    setattr(config_object, attr, attr_type(config_dict[attr]))
                except TypeError:
                    print(f"Type mismatch for attribute {attr}, cannot cast {config_dict[attr]} to {attr_type}")


def create_autogen_memgpt_agent(
    agent_config: dict,
    skip_verify: bool = False,
    interface: bool = None,
    interface_kwargs: dict = {},
    default_auto_reply: Optional[Union[str, Dict, None]] = "",
    is_termination_msg: Optional[Callable[[Dict], bool]] = None,
):
    """
    See AutoGenInterface.__init__ for available options you can pass into
    `interface_kwargs`.  For example, MemGPT's inner monologue and functions are
    off by default so that they are not visible to the other agents. You can
    turn these on by passing in
    ```
    interface_kwargs={
        "debug": True,  # to see all MemGPT activity
        "show_inner_thoughts: True  # to print MemGPT inner thoughts "globally"
                                    # (visible to all AutoGen agents)
    }
    ```
    """
    interface = AutoGenInterface(**interface_kwargs) if interface is None else interface

    config = MemGPTConfig.load()
    llm_config = config.default_llm_config
    embedding_config = config.default_embedding_config

    # Overwrite parts of the LLM and embedding configs that were passed into the config dicts
    update_config_from_dict(llm_config, agent_config)
    update_config_from_dict(embedding_config, agent_config)

    # Create the default user, or load the specified user
    ms = MetadataStore(config)
    if "user_id" not in agent_config:
        user_id = uuid.UUID(config.anon_clientid)
        user = ms.get_user(user_id=user_id)
        if user is None:
            ms.create_user(User(id=user_id))
            user = ms.get_user(user_id=user_id)
            if user is None:
                raise ValueError(f"Failed to create default user {str(user_id)} in database.")
    else:
        user_id = uuid.UUID(agent_config["user_id"])
        user = ms.get_user(user_id=user_id)

    agent_state = AgentState(
        name=agent_config["name"],
        user_id=user_id,
        persona=agent_config["persona"],
        human=agent_config["human"],
        llm_config=llm_config,
        embedding_config=embedding_config,
        preset=agent_config["preset"],
    )
    try:
        memgpt_agent = presets.create_agent_from_preset(
            agent_state=agent_state,
            interface=interface,
            persona_is_file=False,
            human_is_file=False,
        )
        # Save agent in database immediately after writing
        save_agent(agent=memgpt_agent, ms=ms)
    except ValueError as e:
        raise ValueError(f"Failed to create agent from provided information:\n{agent_config}\n\nError: {str(e)}")

    # After creating the agent, we then need to wrap it in a ConversableAgent so that it can be plugged into AutoGen
    autogen_memgpt_agent = MemGPTConversableAgent(
        name=agent_state.name,
        agent=memgpt_agent,
        default_auto_reply=default_auto_reply,
        is_termination_msg=is_termination_msg,
        skip_verify=skip_verify,
    )
    return autogen_memgpt_agent


class MemGPTConversableAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        agent: MemGPTAgent,
        skip_verify=False,
        concat_other_agent_messages=False,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        # TODO: pass in MemGPT config (needed to create DB connections)
    ):
        """A wrapper around a MemGPT agent that implements the AutoGen ConversibleAgent functions

        This allows the MemGPT agent to be used in an AutoGen groupchat
        """
        super().__init__(name, llm_config=False)
        self.agent = agent
        self.skip_verify = skip_verify
        self.concat_other_agent_messages = concat_other_agent_messages
        self.register_reply([Agent, None], MemGPTConversableAgent._generate_reply_for_user_message)
        self.messages_processed_up_to_idx = 0
        self._default_auto_reply = default_auto_reply

        self._is_termination_msg = is_termination_msg if is_termination_msg is not None else (lambda x: x == "TERMINATE")

    def save(self):
        """Save the MemGPT agent to the database"""
        raise NotImplementedError

    def load(self, name: str, type: str, **kwargs):
        raise DeprecationWarning()

        # call load function based on type
        if type == "directory":
            load_directory(name=name, **kwargs)
        elif type == "webpage":
            load_webpage(name=name, **kwargs)
        elif type == "index":
            load_index(name=name, **kwargs)
        elif type == "database":
            load_database(name=name, **kwargs)
        elif type == "vector_database":
            load_vector_database(name=name, **kwargs)
        else:
            raise ValueError(f"Invalid data source type {type}")

    def attach(self, data_source: str):
        raise DeprecationWarning()

        # attach new data
        attach(self.agent.config.name, data_source)

        # update agent config
        self.agent.config.attach_data_source(data_source)

        # reload agent with new data source
        # TODO: @charles we will need to pass in the MemGPT config here to get the DB URIs (not contained in agent)
        self.agent.persistence_manager.archival_memory.storage = StorageConnector.get_archival_storage_connector(
            agent_config=self.agent.config
        )

    def load_and_attach(self, name: str, type: str, force=False, **kwargs):
        raise DeprecationWarning()

        # check if data source already exists
        data_sources = StorageConnector.get_metadata_storage_connector(TableType.DATA_SOURCES).get_all()
        data_sources = [source.name for source in data_sources]
        if name in data_sources and not force:
            print(f"Data source {name} already exists. Use force=True to overwrite.")
            self.attach(name)
        else:
            self.load(name, type, **kwargs)
            self.attach(name)

    def format_other_agent_message(self, msg):
        if "name" in msg:
            user_message = f"{msg['name']}: {msg['content']}"
        else:
            user_message = msg["content"]
        return user_message

    def find_last_user_message(self):
        last_user_message = None
        for msg in self.agent.messages:
            if msg["role"] == "user":
                last_user_message = msg["content"]
        return last_user_message

    def find_new_messages(self, entire_message_list):
        """Extract the subset of messages that's actually new"""
        return entire_message_list[self.messages_processed_up_to_idx :]

    @staticmethod
    def _format_autogen_message(autogen_message):
        # {'content': "...", 'name': '...', 'role': 'user'}
        if not isinstance(autogen_message, dict) or ():
            print(f"Warning: AutoGen message was not a dict -- {autogen_message}")
            user_message = system.package_user_message(autogen_message)
        elif "content" not in autogen_message or "name" not in autogen_message or "name" not in autogen_message:
            print(f"Warning: AutoGen message was missing fields -- {autogen_message}")
            user_message = system.package_user_message(autogen_message)
        else:
            user_message = system.package_user_message(user_message=autogen_message["content"], name=autogen_message["name"])

        return user_message

    def _generate_reply_for_user_message(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        assert isinstance(
            self.agent.interface, AutoGenInterface
        ), f"MemGPT AutoGen Agent is using the wrong interface - {self.agent.interface}"
        self.agent.interface.reset_message_list()

        new_messages = self.find_new_messages(messages)
        new_messages_count = len(new_messages)
        if new_messages_count > 1:
            if self.concat_other_agent_messages:
                # Combine all the other messages into one message
                user_message = "\n".join([self.format_other_agent_message(m) for m in new_messages])
            else:
                # Extend the MemGPT message list with multiple 'user' messages, then push the last one with agent.step()
                self.agent.append_to_messages(new_messages[:-1])
                user_message = new_messages[-1]
        elif new_messages_count == 1:
            user_message = new_messages[0]
        else:
            return True, self._default_auto_reply

        # Package the user message
        # user_message = system.package_user_message(user_message)
        user_message = self._format_autogen_message(user_message)

        # Send a single message into MemGPT
        while True:
            (
                new_messages,
                heartbeat_request,
                function_failed,
                token_warning,
                tokens_accumulated,
            ) = self.agent.step(user_message, first_message=False, skip_verify=self.skip_verify)
            # Skip user inputs if there's a memory warning, function execution failed, or the agent asked for control
            if token_warning:
                user_message = system.get_token_limit_warning()
            elif function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
            elif heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
            else:
                break

        # Stop the conversation
        if self._is_termination_msg(new_messages[-1]["content"]):
            return True, None

        # Pass back to AutoGen the pretty-printed calls MemGPT made to the interface
        pretty_ret = MemGPTConversableAgent.pretty_concat(self.agent.interface.message_list)
        self.messages_processed_up_to_idx += new_messages_count
        return True, pretty_ret

    @staticmethod
    def pretty_concat(messages):
        """AutoGen expects a single response, but MemGPT may take many steps.

        To accommodate AutoGen, concatenate all of MemGPT's steps into one and return as a single message.
        """
        ret = {"role": "assistant", "content": ""}
        lines = []
        for m in messages:
            lines.append(f"{m}")
        ret["content"] = "\n".join(lines)

        # prevent error in LM Studio caused by scenarios where MemGPT didn't say anything
        if ret["content"] in ["", "\n"]:
            ret["content"] = "..."

        return ret
