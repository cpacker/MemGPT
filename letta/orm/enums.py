from enum import Enum


class ToolSourceType(str, Enum):
    """Defines what a tool was derived from"""

    python = "python"
    json = "json"
