__version__ = "0.5.2"

# import clients
from letta.client.client import LocalClient, RESTClient, create_client

# imports for easier access
from letta.schemas.agent import AgentState
from letta.schemas.block import Block
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import JobStatus
from letta.schemas.file import FileMetadata
from letta.schemas.job import Job
from letta.schemas.letta_message import LettaMessage
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import (
    ArchivalMemorySummary,
    BasicBlockMemory,
    ChatMemory,
    Memory,
    RecallMemorySummary,
)
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.organization import Organization
from letta.schemas.passage import Passage
from letta.schemas.source import Source
from letta.schemas.tool import Tool
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
