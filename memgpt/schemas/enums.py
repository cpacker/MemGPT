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

    created = "created", "The job has been created."
    running = "running", "The job is currently running."
    completed = "completed", "The job has been completed."
    failed = "failed", "The job has failed."
    pending = "pending", "The job is pending (has not started running)."


class MessageStreamStatus(str, Enum):
    done_generation = "[DONE_GEN]"
    done_step = "[DONE_STEP]"
    done = "[DONE]"
