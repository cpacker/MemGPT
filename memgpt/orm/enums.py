from enum import Enum


class ToolSourceType(str, Enum):
    """Defines what a tool was derived from"""

    python = "python"
    json = "json"


class JobStatus(str, Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"


class MessageRoleType(str, Enum):
    user = "user"
    system = "system"
