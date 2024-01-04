""" This module contains the data types used by MemGPT. Each data type must include a function to create a DB model. """
import uuid
from abc import abstractmethod
from typing import Optional, List, Dict
import numpy as np


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
