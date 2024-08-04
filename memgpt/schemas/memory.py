from typing import Dict, Optional

from pydantic import BaseModel, Field

from memgpt.schemas.block import Block


class Memory(BaseModel, validate_assignment=True):
    """Represents the in-context memory of the agent"""

    memory: Dict[str, Block] = Field(default_factory=dict, description="Mapping from memory block section to memory block.")

    @classmethod
    def load(cls, state: dict):
        """Load memory from dictionary object"""
        obj = cls()
        for key, value in state.items():
            obj.memory[key] = Block(**value)
        return obj

    def __str__(self) -> str:
        """Representation of the memory in-context"""
        section_strs = []
        for section, module in self.memory.items():
            section_strs.append(f'<{section} characters="{len(module)}/{module.limit}">\n{module.value}\n</{section}>')
        return "\n".join(section_strs)

    def to_dict(self):
        """Convert to dictionary representation"""
        return {key: value.dict() for key, value in self.memory.items()}


class ChatMemory(Memory):

    def __init__(self, persona: str, human: str, limit: int = 2000):
        super().__init__()
        self.memory = {
            "persona": Block(name="persona", value=persona, limit=limit),
            "human": Block(name="human", value=human, limit=limit),
        }

    def core_memory_append(self, name: str, content: str) -> Optional[str]:
        """
        Append to the contents of core memory.

        Args:
            name (str): Section of the memory to be edited (persona or human).
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        self.memory[name].value += "\n" + content
        return None

    def core_memory_replace(self, name: str, old_content: str, new_content: str) -> Optional[str]:
        """
        Replace the contents of core memory. To delete memories, use an empty string for new_content.

        Args:
            name (str): Section of the memory to be edited (persona or human).
            old_content (str): String to replace. Must be an exact match.
            new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        self.memory[name].value = self.memory[name].value.replace(old_content, new_content)
        return None


class UpdateMemory(BaseModel):
    """Update the memory of the agent"""


class ArchivalMemorySummary(BaseModel):
    num_rows: int = Field(..., description="Number of rows in archival memory")


class RecallMemorySummary(BaseModel):
    num_rows: int = Field(..., description="Number of rows in recall memory")
