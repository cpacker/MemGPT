""" This module contains the data types used by MemGPT. Each data type must include a function to create a DB model. """

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from memgpt.constants import (
    DEFAULT_HUMAN,
    DEFAULT_PERSONA,
    DEFAULT_PRESET,
    LLM_MAX_TOKENS,
    MAX_EMBEDDING_DIM,
    TOOL_CALL_ID_MAX_LEN,
)
from memgpt.local_llm.constants import INNER_THOUGHTS_KWARG
from memgpt.prompts import gpt_system
from memgpt.utils import (
    create_uuid_from_string,
    get_human_text,
    get_persona_text,
    get_utc_time,
    is_utc_datetime,
)


class Record:
    """
    Base class for an agent's memory unit. Each memory unit is represented in the database as a single row.
    Memory units are searched over by functions defined in the memory classes
    """

    def __init__(self, id: Optional[uuid.UUID] = None):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"


# This allows type checking to work when you pass a Passage into a function expecting List[Record]
# (just use List[RecordType] instead)
RecordType = TypeVar("RecordType", bound="Record")


class ToolCall(object):
    def __init__(
        self,
        id: str,
        # TODO should we include this? it's fixed to 'function' only (for now) in OAI schema
        # NOTE: called ToolCall.type in official OpenAI schema
        tool_call_type: str,  # only 'function' is supported
        # function: { 'name': ..., 'arguments': ...}
        function: Dict[str, str],
    ):
        self.id = id
        self.tool_call_type = tool_call_type
        self.function = function

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.tool_call_type,
            "function": self.function,
        }


class Message(Record):
    """Representation of a message sent.

    Messages can be:
    - agent->user (role=='agent')
    - user->agent and system->agent (role=='user')
    - or function/tool call returns (role=='function'/'tool').
    """

    def __init__(
        self,
        role: str,
        text: str,
        user_id: Optional[uuid.UUID] = None,
        agent_id: Optional[uuid.UUID] = None,
        model: Optional[str] = None,  # model used to make function call
        name: Optional[str] = None,  # optional participant name
        created_at: Optional[datetime] = None,
        tool_calls: Optional[List[ToolCall]] = None,  # list of tool calls requested
        tool_call_id: Optional[str] = None,
        # tool_call_name: Optional[str] = None,  # not technically OpenAI spec, but it can be helpful to have on-hand
        embedding: Optional[np.ndarray] = None,
        embedding_dim: Optional[int] = None,
        embedding_model: Optional[str] = None,
        id: Optional[uuid.UUID] = None,
    ):
        super().__init__(id)
        self.user_id = user_id
        self.agent_id = agent_id
        self.text = text
        self.model = model  # model name (e.g. gpt-4)
        self.created_at = created_at if created_at is not None else get_utc_time()

        # openai info
        assert role in ["system", "assistant", "user", "tool"]
        self.role = role  # role (agent/user/function)
        self.name = name

        # pad and store embeddings
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        self.embedding = (
            np.pad(embedding, (0, MAX_EMBEDDING_DIM - embedding.shape[0]), mode="constant").tolist() if embedding is not None else None
        )
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        if self.embedding is not None:
            assert self.embedding_dim, f"Must specify embedding_dim if providing an embedding"
            assert self.embedding_model, f"Must specify embedding_model if providing an embedding"
            assert len(self.embedding) == MAX_EMBEDDING_DIM, f"Embedding must be of length {MAX_EMBEDDING_DIM}"

        # tool (i.e. function) call info (optional)

        # if role == "assistant", this MAY be specified
        # if role != "assistant", this must be null
        assert tool_calls is None or isinstance(tool_calls, list)
        self.tool_calls = tool_calls

        # if role == "tool", then this must be specified
        # if role != "tool", this must be null
        if role == "tool":
            assert tool_call_id is not None
        else:
            assert tool_call_id is None
        self.tool_call_id = tool_call_id

    def to_json(self):
        json_message = vars(self)
        if json_message["tool_calls"] is not None:
            json_message["tool_calls"] = [vars(tc) for tc in json_message["tool_calls"]]
        # turn datetime to ISO format
        # also if the created_at is missing a timezone, add UTC
        if not is_utc_datetime(self.created_at):
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        json_message["created_at"] = self.created_at.isoformat()
        return json_message

    @staticmethod
    def dict_to_message(
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        openai_message_dict: dict,
        model: Optional[str] = None,  # model used to make function call
        allow_functions_style: bool = False,  # allow deprecated functions style?
        created_at: Optional[datetime] = None,
    ):
        """Convert a ChatCompletion message object into a Message object (synced to DB)"""

        assert "role" in openai_message_dict, openai_message_dict
        assert "content" in openai_message_dict, openai_message_dict

        # If we're going from deprecated function form
        if openai_message_dict["role"] == "function":
            if not allow_functions_style:
                raise DeprecationWarning(openai_message_dict)
            assert "tool_call_id" in openai_message_dict, openai_message_dict

            # Convert from 'function' response to a 'tool' response
            # NOTE: this does not conventionally include a tool_call_id, it's on the caster to provide it
            return Message(
                created_at=created_at,
                user_id=user_id,
                agent_id=agent_id,
                model=model,
                # standard fields expected in an OpenAI ChatCompletion message object
                role="tool",  # NOTE
                text=openai_message_dict["content"],
                name=openai_message_dict["name"] if "name" in openai_message_dict else None,
                tool_calls=openai_message_dict["tool_calls"] if "tool_calls" in openai_message_dict else None,
                tool_call_id=openai_message_dict["tool_call_id"] if "tool_call_id" in openai_message_dict else None,
            )

        elif "function_call" in openai_message_dict and openai_message_dict["function_call"] is not None:
            if not allow_functions_style:
                raise DeprecationWarning(openai_message_dict)
            assert openai_message_dict["role"] == "assistant", openai_message_dict
            assert "tool_call_id" in openai_message_dict, openai_message_dict

            # Convert a function_call (from an assistant message) into a tool_call
            # NOTE: this does not conventionally include a tool_call_id (ToolCall.id), it's on the caster to provide it
            tool_calls = [
                ToolCall(
                    id=openai_message_dict["tool_call_id"],  # NOTE: unconventional source, not to spec
                    tool_call_type="function",
                    function={
                        "name": openai_message_dict["function_call"]["name"],
                        "arguments": openai_message_dict["function_call"]["arguments"],
                    },
                )
            ]

            return Message(
                created_at=created_at,
                user_id=user_id,
                agent_id=agent_id,
                model=model,
                # standard fields expected in an OpenAI ChatCompletion message object
                role=openai_message_dict["role"],
                text=openai_message_dict["content"],
                name=openai_message_dict["name"] if "name" in openai_message_dict else None,
                tool_calls=tool_calls,
                tool_call_id=None,  # NOTE: None, since this field is only non-null for role=='tool'
            )

        else:
            # Basic sanity check
            if openai_message_dict["role"] == "tool":
                assert "tool_call_id" in openai_message_dict and openai_message_dict["tool_call_id"] is not None, openai_message_dict
            else:
                if "tool_call_id" in openai_message_dict:
                    assert openai_message_dict["tool_call_id"] is None, openai_message_dict

            if "tool_calls" in openai_message_dict and openai_message_dict["tool_calls"] is not None:
                assert openai_message_dict["role"] == "assistant", openai_message_dict

                tool_calls = [
                    ToolCall(id=tool_call["id"], tool_call_type=tool_call["type"], function=tool_call["function"])
                    for tool_call in openai_message_dict["tool_calls"]
                ]
            else:
                tool_calls = None

            # If we're going from tool-call style
            return Message(
                created_at=created_at,
                user_id=user_id,
                agent_id=agent_id,
                model=model,
                # standard fields expected in an OpenAI ChatCompletion message object
                role=openai_message_dict["role"],
                text=openai_message_dict["content"],
                name=openai_message_dict["name"] if "name" in openai_message_dict else None,
                tool_calls=tool_calls,
                tool_call_id=openai_message_dict["tool_call_id"] if "tool_call_id" in openai_message_dict else None,
            )

    def to_openai_dict_search_results(self, max_tool_id_length=TOOL_CALL_ID_MAX_LEN) -> dict:
        result_json = self.to_openai_dict()
        search_result_json = {"timestamp": self.created_at, "message": {"content": result_json["content"], "role": result_json["role"]}}
        return search_result_json

    def to_openai_dict(self, max_tool_id_length=TOOL_CALL_ID_MAX_LEN) -> dict:
        """Go from Message class to ChatCompletion message object"""

        # TODO change to pydantic casting, eg `return SystemMessageModel(self)`

        if self.role == "system":
            assert all([v is not None for v in [self.role]]), vars(self)
            openai_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional field, do not include if null
            if self.name is not None:
                openai_message["name"] = self.name

        elif self.role == "user":
            assert all([v is not None for v in [self.text, self.role]]), vars(self)
            openai_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional field, do not include if null
            if self.name is not None:
                openai_message["name"] = self.name

        elif self.role == "assistant":
            assert self.tool_calls is not None or self.text is not None
            openai_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional fields, do not include if null
            if self.name is not None:
                openai_message["name"] = self.name
            if self.tool_calls is not None:
                openai_message["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
                if max_tool_id_length:
                    for tool_call_dict in openai_message["tool_calls"]:
                        tool_call_dict["id"] = tool_call_dict["id"][:max_tool_id_length]

        elif self.role == "tool":
            assert all([v is not None for v in [self.role, self.tool_call_id]]), vars(self)
            openai_message = {
                "content": self.text,
                "role": self.role,
                "tool_call_id": self.tool_call_id[:max_tool_id_length] if max_tool_id_length else self.tool_call_id,
            }

        else:
            raise ValueError(self.role)

        return openai_message

    def to_anthropic_dict(self, inner_thoughts_xml_tag="thinking") -> dict:
        # raise NotImplementedError

        def add_xml_tag(string: str, xml_tag: Optional[str]):
            # NOTE: Anthropic docs recommends using <thinking> tag when using CoT + tool use
            return f"<{xml_tag}>{string}</{xml_tag}" if xml_tag else string

        if self.role == "system":
            raise ValueError(f"Anthropic 'system' role not supported")

        elif self.role == "user":
            assert all([v is not None for v in [self.text, self.role]]), vars(self)
            anthropic_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional field, do not include if null
            if self.name is not None:
                anthropic_message["name"] = self.name

        elif self.role == "assistant":
            assert self.tool_calls is not None or self.text is not None
            anthropic_message = {
                "role": self.role,
            }
            content = []
            if self.text is not None:
                content.append(
                    {
                        "type": "text",
                        "text": add_xml_tag(string=self.text, xml_tag=inner_thoughts_xml_tag),
                    }
                )
            if self.tool_calls is not None:
                for tool_call in self.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function["name"],
                            "input": json.loads(tool_call.function["arguments"]),
                        }
                    )

            # If the only content was text, unpack it back into a singleton
            # TODO
            anthropic_message["content"] = content

            # Optional fields, do not include if null
            if self.name is not None:
                anthropic_message["name"] = self.name

        elif self.role == "tool":
            # NOTE: Anthropic uses role "user" for "tool" responses
            assert all([v is not None for v in [self.role, self.tool_call_id]]), vars(self)
            anthropic_message = {
                "role": "user",  # NOTE: diff
                "content": [
                    # TODO support error types etc
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id,
                        "content": self.text,
                    }
                ],
            }

        else:
            raise ValueError(self.role)

        return anthropic_message

    def to_google_ai_dict(self, put_inner_thoughts_in_kwargs: bool = True) -> dict:
        """Go from Message class to Google AI REST message object

        type Content: https://ai.google.dev/api/rest/v1/Content / https://ai.google.dev/api/rest/v1beta/Content
            parts[]: Part
            role: str ('user' or 'model')
        """
        if self.role != "tool" and self.name is not None:
            raise UserWarning(f"Using Google AI with non-null 'name' field ({self.name}) not yet supported.")

        if self.role == "system":
            # NOTE: Gemini API doesn't have a 'system' role, use 'user' instead
            # https://www.reddit.com/r/Bard/comments/1b90i8o/does_gemini_have_a_system_prompt_option_while/
            google_ai_message = {
                "role": "user",  # NOTE: no 'system'
                "parts": [{"text": self.text}],
            }

        elif self.role == "user":
            assert all([v is not None for v in [self.text, self.role]]), vars(self)
            google_ai_message = {
                "role": "user",
                "parts": [{"text": self.text}],
            }

        elif self.role == "assistant":
            assert self.tool_calls is not None or self.text is not None
            google_ai_message = {
                "role": "model",  # NOTE: different
            }

            # NOTE: Google AI API doesn't allow non-null content + function call
            # To get around this, just two a two part message, inner thoughts first then
            parts = []
            if not put_inner_thoughts_in_kwargs and self.text is not None:
                # NOTE: ideally we do multi-part for CoT / inner thoughts + function call, but Google AI API doesn't allow it
                raise NotImplementedError
                parts.append({"text": self.text})

            if self.tool_calls is not None:
                # NOTE: implied support for multiple calls
                for tool_call in self.tool_calls:
                    function_name = tool_call.function["name"]
                    function_args = tool_call.function["arguments"]
                    try:
                        # NOTE: Google AI wants actual JSON objects, not strings
                        function_args = json.loads(function_args)
                    except:
                        raise UserWarning(f"Failed to parse JSON function args: {function_args}")
                        function_args = {"args": function_args}

                    if put_inner_thoughts_in_kwargs and self.text is not None:
                        assert "inner_thoughts" not in function_args, function_args
                        assert len(self.tool_calls) == 1
                        function_args[INNER_THOUGHTS_KWARG] = self.text

                    parts.append(
                        {
                            "functionCall": {
                                "name": function_name,
                                "args": function_args,
                            }
                        }
                    )
            else:
                assert self.text is not None
                parts.append({"text": self.text})
            google_ai_message["parts"] = parts

        elif self.role == "tool":
            # NOTE: Significantly different tool calling format, more similar to function calling format
            assert all([v is not None for v in [self.role, self.tool_call_id]]), vars(self)

            if self.name is None:
                raise UserWarning(f"Couldn't find function name on tool call, defaulting to tool ID instead.")
                function_name = self.tool_call_id
            else:
                function_name = self.name

            # NOTE: Google AI API wants the function response as JSON only, no string
            try:
                function_response = json.loads(self.text)
            except:
                function_response = {"function_response": self.text}

            google_ai_message = {
                "role": "function",
                "parts": [
                    {
                        "functionResponse": {
                            "name": function_name,
                            "response": {
                                "name": function_name,  # NOTE: name twice... why?
                                "content": function_response,
                            },
                        }
                    }
                ],
            }

        else:
            raise ValueError(self.role)

        return google_ai_message

    def to_cohere_dict(
        self,
        function_call_role: Optional[str] = "SYSTEM",
        function_call_prefix: Optional[str] = "[CHATBOT called function]",
        function_response_role: Optional[str] = "SYSTEM",
        function_response_prefix: Optional[str] = "[CHATBOT function returned]",
        inner_thoughts_as_kwarg: Optional[bool] = False,
    ) -> List[dict]:
        """Cohere chat_history dicts only have 'role' and 'message' fields

        NOTE: returns a list of dicts so that we can convert:
          assistant [cot]: "I'll send a message"
          assistant [func]: send_message("hi")
          tool: {'status': 'OK'}
        to:
          CHATBOT.text: "I'll send a message"
          SYSTEM.text: [CHATBOT called function] send_message("hi")
          SYSTEM.text: [CHATBOT function returned] {'status': 'OK'}

        TODO: update this prompt style once guidance from Cohere on
        embedded function calls in multi-turn conversation become more clear
        """

        if self.role == "system":
            """
            The chat_history parameter should not be used for SYSTEM messages in most cases.
            Instead, to add a SYSTEM role message at the beginning of a conversation, the preamble parameter should be used.
            """
            raise UserWarning(f"role 'system' messages should go in 'preamble' field for Cohere API")

        elif self.role == "user":
            assert all([v is not None for v in [self.text, self.role]]), vars(self)
            cohere_message = [
                {
                    "role": "USER",
                    "message": self.text,
                }
            ]

        elif self.role == "assistant":
            # NOTE: we may break this into two message - an inner thought and a function call
            # Optionally, we could just make this a function call with the inner thought inside
            assert self.tool_calls is not None or self.text is not None

            if self.text and self.tool_calls:
                if inner_thoughts_as_kwarg:
                    raise NotImplementedError
                cohere_message = [
                    {
                        "role": "CHATBOT",
                        "message": self.text,
                    },
                ]
                for tc in self.tool_calls:
                    # TODO better way to pack?
                    # function_call_text = json.dumps(tc.to_dict())
                    function_name = tc.function["name"]
                    function_args = json.loads(tc.function["arguments"])
                    function_args_str = ",".join([f"{k}={v}" for k, v in function_args.items()])
                    function_call_text = f"{function_name}({function_args_str})"
                    cohere_message.append(
                        {
                            "role": function_call_role,
                            "message": f"{function_call_prefix} {function_call_text}",
                        }
                    )
            elif not self.text and self.tool_calls:
                cohere_message = []
                for tc in self.tool_calls:
                    # TODO better way to pack?
                    function_call_text = json.dumps(tc.to_dict())
                    cohere_message.append(
                        {
                            "role": function_call_role,
                            "message": f"{function_call_prefix} {function_call_text}",
                        }
                    )
            elif self.text and not self.tool_calls:
                cohere_message = [
                    {
                        "role": "CHATBOT",
                        "message": self.text,
                    }
                ]
            else:
                raise ValueError("Message does not have content nor tool_calls")

        elif self.role == "tool":
            assert all([v is not None for v in [self.role, self.tool_call_id]]), vars(self)
            function_response_text = self.text
            cohere_message = [
                {
                    "role": function_response_role,
                    "message": f"{function_response_prefix} {function_response_text}",
                }
            ]

        else:
            raise ValueError(self.role)

        return cohere_message


class Document(Record):
    """A document represent a document loaded into MemGPT, which is broken down into passages."""

    def __init__(self, user_id: uuid.UUID, text: str, data_source: str, id: Optional[uuid.UUID] = None, metadata: Optional[Dict] = {}):
        if id is None:
            # by default, generate ID as a hash of the text (avoid duplicates)
            self.id = create_uuid_from_string("".join([text, str(user_id)]))
        else:
            self.id = id
        super().__init__(id)
        self.user_id = user_id
        self.text = text
        self.data_source = data_source
        self.metadata = metadata
        # TODO: add optional embedding?


class Passage(Record):
    """A passage is a single unit of memory, and a standard format accross all storage backends.

    It is a string of text with an assoidciated embedding.
    """

    def __init__(
        self,
        text: str,
        user_id: Optional[uuid.UUID] = None,
        agent_id: Optional[uuid.UUID] = None,  # set if contained in agent memory
        embedding: Optional[np.ndarray] = None,
        embedding_dim: Optional[int] = None,
        embedding_model: Optional[str] = None,
        data_source: Optional[str] = None,  # None if created by agent
        doc_id: Optional[uuid.UUID] = None,
        id: Optional[uuid.UUID] = None,
        metadata_: Optional[dict] = {},
        created_at: Optional[datetime] = None,
    ):
        if id is None:
            # by default, generate ID as a hash of the text (avoid duplicates)
            # TODO: use source-id instead?
            if agent_id:
                self.id = create_uuid_from_string("".join([text, str(agent_id), str(user_id)]))
            else:
                self.id = create_uuid_from_string("".join([text, str(user_id)]))
        else:
            self.id = id
        super().__init__(self.id)
        self.user_id = user_id
        self.agent_id = agent_id
        self.text = text
        self.data_source = data_source
        self.doc_id = doc_id
        self.metadata_ = metadata_

        # pad and store embeddings
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        self.embedding = (
            np.pad(embedding, (0, MAX_EMBEDDING_DIM - embedding.shape[0]), mode="constant").tolist() if embedding is not None else None
        )
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        self.created_at = created_at if created_at is not None else get_utc_time()

        if self.embedding is not None:
            assert self.embedding_dim, f"Must specify embedding_dim if providing an embedding"
            assert self.embedding_model, f"Must specify embedding_model if providing an embedding"
            assert len(self.embedding) == MAX_EMBEDDING_DIM, f"Embedding must be of length {MAX_EMBEDDING_DIM}"

        assert isinstance(self.user_id, uuid.UUID), f"UUID {self.user_id} must be a UUID type"
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"
        assert not agent_id or isinstance(self.agent_id, uuid.UUID), f"UUID {self.agent_id} must be a UUID type"
        assert not doc_id or isinstance(self.doc_id, uuid.UUID), f"UUID {self.doc_id} must be a UUID type"


class LLMConfig:
    def __init__(
        self,
        model: Optional[str] = None,
        model_endpoint_type: Optional[str] = None,
        model_endpoint: Optional[str] = None,
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


class EmbeddingConfig:
    def __init__(
        self,
        embedding_endpoint_type: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_chunk_size: Optional[int] = 300,
    ):
        self.embedding_endpoint_type = embedding_endpoint_type
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_chunk_size = embedding_chunk_size

        # fields cannot be set to None
        assert self.embedding_endpoint_type
        assert self.embedding_dim
        assert self.embedding_chunk_size


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
        # name: str,
        id: Optional[uuid.UUID] = None,
        default_agent=None,
        # other
        policies_accepted=False,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"

        self.default_agent = default_agent

        # misc
        self.policies_accepted = policies_accepted


class AgentState:
    def __init__(
        self,
        name: str,
        user_id: uuid.UUID,
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
        created_at: Optional[datetime] = None,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"
        assert isinstance(user_id, uuid.UUID), f"UUID {user_id} must be a UUID type"

        # TODO(swooders) we need to handle the case where name is None here
        # in AgentConfig we autogenerate a name, not sure what the correct thing w/ DBs is, what about NounAdjective combos? Like giphy does? BoredGiraffe etc
        self.name = name
        self.user_id = user_id
        self.preset = preset
        # The INITIAL values of the persona and human
        # The values inside self.state['persona'], self.state['human'] are the CURRENT values
        self.persona = persona
        self.human = human

        self.llm_config = llm_config
        self.embedding_config = embedding_config

        self.created_at = created_at if created_at is not None else get_utc_time()

        # state
        self.state = {} if not state else state


class Source:
    def __init__(
        self,
        user_id: uuid.UUID,
        name: str,
        description: Optional[str] = None,
        created_at: Optional[datetime] = None,
        id: Optional[uuid.UUID] = None,
        # embedding info
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"
        assert isinstance(user_id, uuid.UUID), f"UUID {user_id} must be a UUID type"

        self.name = name
        self.user_id = user_id
        self.description = description
        self.created_at = created_at if created_at is not None else get_utc_time()

        # embedding info (optional)
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model


class Token:
    def __init__(
        self,
        user_id: uuid.UUID,
        token: str,
        name: Optional[str] = None,
        id: Optional[uuid.UUID] = None,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"
        assert isinstance(user_id, uuid.UUID), f"UUID {user_id} must be a UUID type"

        self.token = token
        self.user_id = user_id
        self.name = name


class Preset(BaseModel):
    name: str = Field(..., description="The name of the preset.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the preset was created.")
    system: str = Field(
        gpt_system.get_system_text(DEFAULT_PRESET), description="The system prompt of the preset."
    )  # default system prompt is same as default preset name
    # system_name: Optional[str] = Field(None, description="The name of the system prompt of the preset.")
    persona: str = Field(default=get_persona_text(DEFAULT_PERSONA), description="The persona of the preset.")
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human: str = Field(default=get_human_text(DEFAULT_HUMAN), description="The human of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")
    functions_schema: List[Dict] = Field(..., description="The functions schema of the preset.")
    # functions: List[str] = Field(..., description="The functions of the preset.") # TODO: convert to ID
    # sources: List[str] = Field(..., description="The sources of the preset.") # TODO: convert to ID

    @staticmethod
    def clone(preset_obj: "Preset", new_name_suffix: str = None) -> "Preset":
        """
        Takes a Preset object and an optional new name suffix as input,
        creates a clone of the given Preset object with a new ID and an optional new name,
        and returns the new Preset object.
        """
        new_preset = preset_obj.model_copy()
        new_preset.id = uuid.uuid4()
        if new_name_suffix:
            new_preset.name = f"{preset_obj.name}_{new_name_suffix}"
        else:
            new_preset.name = f"{preset_obj.name}_{str(uuid.uuid4())[:8]}"
        return new_preset


class Function(BaseModel):
    name: str = Field(..., description="The name of the function.")
    id: uuid.UUID = Field(..., description="The unique identifier of the function.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the function.")
    # TODO: figure out how represent functions
