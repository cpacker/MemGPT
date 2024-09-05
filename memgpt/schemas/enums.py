from enum import Enum


class MessageRole(str, Enum):
    assistant = "assistant"
    user = "user"
    tool = "tool"
    function = "function"
    system = "system"


class OptionState(str, Enum):
    """Useful for kwargs that are bool + default option"""

    YES = "yes"
    NO = "no"
    DEFAULT = "default"


class JobStatus(str, Enum):
    """
    Status of the job.
    """

    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    pending = "pending"


class MessageStreamStatus(str, Enum):
    done_generation = "[DONE_GEN]"
    done_step = "[DONE_STEP]"
    done = "[DONE]"
