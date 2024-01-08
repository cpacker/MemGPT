""" This module contains the data types used by MemGPT. Each data type must include a function to create a DB model. """
import uuid
from datetime import datetime
from abc import abstractmethod
from typing import Optional, List, Dict
import numpy as np

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_MEMGPT_MODEL, DEFAULT_PERSONA, DEFAULT_PRESET, LLM_MAX_TOKENS
from memgpt.utils import get_local_time, format_datetime

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
        name: Optional[str] = None,  # optional participant name
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
        self.name = name

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


class LLMConfig:
    def __init__(
        self,
        model: Optional[str] = "gpt-4",
        model_endpoint_type: Optional[str] = "openai",
        model_endpoint: Optional[str] = "https://api.openai.com/v1",
        model_wrapper: Optional[str] = None,
        context_window: Optional[int] = None,
    ):
        self.model = model
        self.model_endpoint_type = model_endpoint_type
        self.model_endpoint = model_endpoint
        self.model_wrapper = model_wrapper
        self.context_window = context_window

        if context_window is None:
            self.context_window = LLM_MAX_TOKENS[self.model] if self.model in LLM_MAX_TOKENS else LLM_MAX_TOKENS["DEFAULT"]
        else:
            self.context_window = context_window


class OpenAILLMConfig(LLMConfig):
    def __init__(self, openai_key, **kwargs):
        super().__init__(**kwargs)
        self.openai_key = openai_key


class AzureLLMConfig(LLMConfig):
    def __init__(
        self,
        azure_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.azure_key = azure_key
        self.azure_endpoint = azure_endpoint
        self.azure_version = azure_version
        self.azure_deployment = azure_deployment


class EmbeddingConfig:
    def __init__(
        self,
        embedding_endpoint_type: Optional[str] = "local",
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = 384,
        embedding_chunk_size: Optional[int] = 300,
        # openai-only
        openai_key: Optional[str] = None,
        # azure-only
        azure_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        self.embedding_endpoint_type = embedding_endpoint_type
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_chunk_size = embedding_chunk_size

        # openai
        self.openai_key = openai_key

        # azure
        self.azure_key = azure_key
        self.azure_endpoint = azure_endpoint
        self.azure_version = azure_version
        self.azure_deployment = azure_deployment


class OpenAIEmbeddingConfig(EmbeddingConfig):
    def __init__(self, openai_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.openai_key = openai_key


class AzureEmbeddingConfig(EmbeddingConfig):
    def __init__(
        self,
        azure_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.azure_key = azure_key
        self.azure_endpoint = azure_endpoint
        self.azure_version = azure_version
        self.azure_deployment = azure_deployment


class User:

    """Defines user and default configurations"""

    # TODO: make sure to encrypt/decrypt keys before storing in DB

    def __init__(
        self,
        id: Optional[uuid.UUID] = None,
        default_preset=DEFAULT_PRESET,
        default_persona=DEFAULT_PERSONA,
        default_human=DEFAULT_HUMAN,
        default_agent=None,
        default_llm_config: Optional[LLMConfig] = None,  # defaults: llm model
        default_embedding_config: Optional[EmbeddingConfig] = None,  # defaults: embeddings
        # azure information
        azure_key=None,
        azure_endpoint=None,
        azure_version=None,
        azure_deployment=None,
        # openai information
        openai_key=None,
        # other
        policies_accepted=False,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        self.default_preset = default_preset
        self.default_persona = default_persona
        self.default_human = default_human
        self.default_agent = default_agent

        # model defaults
        self.default_llm_config = default_llm_config if default_llm_config is not None else LLMConfig()
        self.default_embedding_config = default_embedding_config if default_embedding_config is not None else EmbeddingConfig()

        # azure information
        # TODO: split this up accross model config and embedding config?
        self.azure_key = azure_key
        self.azure_endpoint = azure_endpoint
        self.azure_version = azure_version
        self.azure_deployment = azure_deployment

        # openai information
        self.openai_key = openai_key

        # set default embedding config
        if default_embedding_config is None:
            if self.openai_key:
                self.default_embedding_config = OpenAIEmbeddingConfig(
                    openai_key=self.openai_key,
                    embedding_endpoint_type="openai",
                    embedding_endpoint="https://api.openai.com/v1",
                    embedding_dim=1536,
                )
            elif self.azure_key:
                self.default_embedding_config = AzureEmbeddingConfig(
                    azure_key=self.azure_key,
                    azure_endpoint=self.azure_endpoint,
                    azure_version=self.azure_version,
                    azure_deployment=self.azure_deployment,
                    embedding_endpoint_type="azure",
                    embedding_endpoint="https://api.openai.com/v1",
                    embedding_dim=1536,
                )
            else:
                # memgpt hosted
                self.default_embedding_config = EmbeddingConfig(
                    embedding_endpoint_type="hugging-face",
                    embedding_endpoint="https://embeddings.memgpt.ai",
                    embedding_model="BAAI/bge-large-en-v1.5",
                    embedding_dim=1024,
                    embedding_chunk_size=300,
                )

        # set default LLM config
        if default_llm_config is None:
            if self.openai_key:
                self.default_llm_config = OpenAILLMConfig(
                    openai_key=self.openai_key,
                    model="gpt-4",
                    model_endpoint_type="openai",
                    model_endpoint="https://api.openai.com/v1",
                    model_wrapper=None,
                    context_window=LLM_MAX_TOKENS["gpt-4"],
                )
            elif self.azure_key:
                self.default_llm_config = AzureLLMConfig(
                    azure_key=self.azure_key,
                    azure_endpoint=self.azure_endpoint,
                    azure_version=self.azure_version,
                    azure_deployment=self.azure_deployment,
                    model="gpt-4",
                    model_endpoint_type="azure",
                    model_endpoint="https://api.openai.com/v1",
                    model_wrapper=None,
                    context_window=LLM_MAX_TOKENS["gpt-4"],
                )
            else:
                # memgpt hosted
                self.default_llm_config = LLMConfig(
                    model="ehartford/dolphin-2.5-mixtral-8x7b",
                    model_endpoint_type="vllm",
                    model_endpoint="https://api.memgpt.ai",
                    model_wrapper="chatml",
                    context_window=16384,
                )

        # misc
        self.policies_accepted = policies_accepted


class AgentState:
    def __init__(
        self,
        name: str,
        user_id: str,
        persona: str,  # the filename where the persona was originally sourced from
        human: str,  # the filename where the human was originally sourced from
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        preset: str,
        # (in-context) state contains:
        # persona: str  # the current persona text
        # human: str  # the current human text
        # system: str,  # system prompt (not required if initializing with a preset)
        # functions: dict,  # schema definitions ONLY (function code linked at runtime)
        # messages: List[dict],  # in-context messages
        id: Optional[uuid.UUID] = None,
        state: Optional[dict] = None,
        created_at: Optional[str] = None,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        # TODO(swooders) we need to handle the case where name is None here
        # in AgentConfig we autogenerate a name, not sure what the correct thing w/ DBs is, what about NounAdjective combos? Like giphy does? BoredGiraffe etc
        self.name = name
        self.user_id = user_id
        self.preset = preset
        self.persona = persona
        self.human = human

        self.llm_config = llm_config
        self.embedding_config = embedding_config

        self.created_at = created_at if created_at is not None else datetime.now()

        # state
        self.state = state

    # def __eq__(self, other):
    #     if not isinstance(other, AgentState):
    #         # return False
    #         return NotImplemented

    #     return (
    #         self.name == other.name
    #         and self.user_id == other.user_id
    #         and self.persona == other.persona
    #         and self.human == other.human
    #         and vars(self.llm_config) == vars(other.llm_config)
    #         and vars(self.embedding_config) == vars(other.embedding_config)
    #         and self.preset == other.preset
    #         and self.state == other.state
    #     )

    # def __dict__(self):
    #    return {
    #        "id": self.id,
    #        "name": self.name,
    #        "user_id": self.user_id,
    #        "preset": self.preset,
    #        "persona": self.persona,
    #        "human": self.human,
    #        "llm_config": self.llm_config,
    #        "embedding_config": self.embedding_config,
    #        "created_at": format_datetime(self.created_at),
    #        "state": self.state,
    #    }


class Source:
    def __init__(
        self,
        user_id: str,
        name: str,
        created_at: Optional[str] = None,
        id: Optional[uuid.UUID] = None,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        self.name = name
        self.user_id = user_id
        self.created_at = created_at
