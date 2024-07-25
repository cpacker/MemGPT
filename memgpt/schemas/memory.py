from typing import Dict

from pydantic import BaseModel, Field

from memgpt.schemas.block import Block


class Memory(BaseModel):
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
