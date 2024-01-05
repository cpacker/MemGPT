""" This module contains the data types used by MemGPT. Each data type must include a function to create a DB model. """
import uuid
from abc import abstractmethod
from typing import Optional, List, Dict
import numpy as np

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_PRESET, LLM_MAX_TOKENS

# Defining schema objects:
# Note: user/agent can borrow from MemGPTConfig/AgentConfig classes


class Record:
    """
    Base class for an agent's memory unit. Each memory unit is represented in the database as a single row.
    Memory units are searched over by functions defined in the memory classes
    """

    def __init__(self, id: Optional[str] = None):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"


class ToolCall(object):
    def __init__(
        self,
        id: str,
        # TODO should we include this? it's fixed to 'function' only (for now) in OAI schema
        tool_call_type: str,  # only 'function' is supported
        # function: { 'name': ..., 'arguments': ...}
        function: Dict[str, str],
    ):
        self.id = id
        self.tool_call_type = tool_call_type
        self.function = function


class Message(Record):
    """Representation of a message sent.

    Messages can be:
    - agent->user (role=='agent')
    - user->agent and system->agent (role=='user')
    - or function/tool call returns (role=='function'/'tool').
    """

    def __init__(
        self,
        user_id: str,
        agent_id: str,
        role: str,
        text: str,
        model: str,  # model used to make function call
        user: Optional[str] = None,  # optional participant name
        created_at: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,  # list of tool calls requested
        tool_call_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        id: Optional[str] = None,
    ):
        super().__init__(id)
        self.user_id = user_id
        self.agent_id = agent_id
        self.text = text
        self.model = model  # model name (e.g. gpt-4)
        self.created_at = created_at

        # openai info
        self.role = role  # role (agent/user/function)
        self.user = user

        # tool (i.e. function) call info (optional)

        # if role == "assistant", this MAY be specified
        # if role != "assistant", this must be null
        self.tool_calls = tool_calls

        # if role == "tool", then this must be specified
        # if role != "tool", this must be null
        self.tool_call_id = tool_call_id

        # embedding (optional)
        self.embedding = embedding

    # def __repr__(self):
    #    pass


class Document(Record):
    """A document represent a document loaded into MemGPT, which is broken down into passages."""

    def __init__(self, user_id: str, text: str, data_source: str, document_id: Optional[str] = None):
        super().__init__(id)
        self.user_id = user_id
        self.text = text
        self.document_id = document_id
        self.data_source = data_source
        # TODO: add optional embedding?

    # def __repr__(self) -> str:
    #    pass


class Passage(Record):
    """A passage is a single unit of memory, and a standard format accross all storage backends.

    It is a string of text with an assoidciated embedding.
    """

    def __init__(
        self,
        user_id: str,
        text: str,
        agent_id: Optional[str] = None,  # set if contained in agent memory
        embedding: Optional[np.ndarray] = None,
        data_source: Optional[str] = None,  # None if created by agent
        doc_id: Optional[str] = None,
        id: Optional[str] = None,
        metadata: Optional[dict] = {},
    ):
        super().__init__(id)
        self.user_id = user_id
        self.agent_id = agent_id
        self.text = text
        self.data_source = data_source
        self.embedding = embedding
        self.doc_id = doc_id
        self.metadata = metadata

    # def __repr__(self):
    #    pass


class Source(Record):
    def __init__(
        self,
        user_id: str,
        name: str,
        created_at: Optional[str] = None,
        id: Optional[str] = None,
    ):
        super().__init__(id)
        self.name = name
        self.user_id = user_id
        self.created_at = created_at


class User:

    """Defines user and default configurations"""

    # TODO: make sure to encrypt/decrypt keys before storing in DB

    def __init__(
        self,
        default_preset=DEFAULT_PRESET,
        default_persona=DEFAULT_PERSONA,
        default_human=DEFAULT_HUMAN,
        default_agent=None,
        # defaults: llm model
        default_model=DEFAULT_MEMGPT_MODEL,
        default_model_endpoint_type="openai",
        default_model_endpoint="https://api.openai.com/v1",
        default_model_wrapper=None,
        default_context_window=None,
        # defaults: embeddings
        default_embedding_endpoint_type="openai",
        default_embedding_endpoint=None,
        default_embedding_model=None,
        default_embedding_dim=None,
        default_embedding_chunk_size=None,
        # azure information
        azure_key=None,
        azure_endpoint=None,
        azure_version=None,
        azure_deployment=None,
        anon_clientid=None,
        # openai information
        openai_key=None,
        # other
        memgpt_version=None,
        policies_accepted=False,
    ):
        self.default_preset = default_preset
        self.default_persona = default_persona
        self.default_human = default_human
        self.default_agent = default_agent

        # defaults: llm model
        self.default_model = default_model
        self.default_model_endpoint_type = default_model_endpoint_type
        self.default_model_endpoint = default_model_endpoint
        self.default_model_wrapper = default_model_wrapper
        if default_context_window is None:
            self.default_context_window = (
                LLM_MAX_TOKENS[self.default_model] if self.default_model in LLM_MAX_TOKENS else LLM_MAX_TOKENS["DEFAULT"]
            )
        else:
            self.default_context_window = default_context_window

        # defaults: embeddings
        self.default_embedding_endpoint_type = default_embedding_endpoint_type
        self.default_embedding_endpoint = default_embedding_endpoint
        self.default_embedding_model = default_embedding_model
        self.default_embedding_dim = default_embedding_dim
        self.default_embedding_chunk_size = default_embedding_chunk_size

        # azure information
        self.azure_key = azure_key
        self.azure_endpoint = azure_endpoint
        self.azure_version = azure_version
        self.azure_deployment = azure_deployment

        # openai information
        self.openai_key = openai_key

        # misc
        self.memgpt_version = memgpt_version

        # TODO: generate
        self.anon_clientid = anon_clientid
        self.policies_accepted = policies_accepted


class AgentState(Record):
    def __init__(
        self,
        name: str,
        persona_file: str,  # the filename where the persona was originally sourced from
        human_file: str,  # the filename where the human was originally sourced from
        # (in-context) state contains:
        # persona: str  # the current persona text
        # human: str  # the current human text
        # system: str,  # system prompt (not required if initializing with a preset)
        # functions: dict,  # schema definitions ONLY (function code linked at runtime)
        # messages: List[dict],  # in-context messages
        state: Optional[dict] = None,
        # model info
        model: Optional[str] = None,
        model_endpoint_type: Optional[str] = None,
        model_endpoint: Optional[str] = None,
        model_wrapper: Optional[str] = None,
        context_window: Optional[int] = None,  # TODO(swooders) I don't think this should be optional
        # embedding info
        embedding_endpoint_type: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,  # TODO(swooders) similarly these probably should not be optional
        embedding_chunk_size: Optional[int] = None,
        # other
        preset: Optional[str] = None,
        data_sources: Optional[list] = None,
        create_time: Optional[str] = None,
        memgpt_version: Optional[str] = None,
    ):
        # TODO(swooders) we need to handle the case where name is None here
        # in AgentConfig we autogenerate a name, not sure what the correct thing w/ DBs is, what about NounAdjective combos? Like giphy does? BoredGiraffe etc
        self.name = name
        self.persona_file = DEFAULT_PERSONA if persona_file is None else persona_file
        self.human_file = DEFAULT_HUMAN if persona_file is None else human_file

        assert context_window is not None

        # model info
        self.model = model
        self.model_endpoint_type = model_endpoint_type
        self.model_endpoint = model_endpoint
        self.model_wrapper = model_wrapper
        self.context_window = context_window

        # embedding info
        self.embedding_endpoint_type = embedding_endpoint_type
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_chunk_size = embedding_chunk_size

        # other
        # NOTE: preset is only used to determine an initial combination of system message + functions (can be None)
        self.preset = (
            DEFAULT_PRESET if preset is None else preset
        )  # TODO(swooders) we should probably allow this to be None? what if someone wants to create w/o a preset?
        self.data_sources = data_sources
        self.create_time = create_time
        self.memgpt_version = memgpt_version

        # state
        self.state = state
