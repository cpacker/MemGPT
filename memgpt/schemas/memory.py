from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from memgpt.schemas.block import Block


class Memory(BaseModel, validate_assignment=True):
    """Represents the in-context memory of the agent"""

    # Private variable to avoid assignments with incorrect types
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

    def to_flat_dict(self):
        """Convert to a dictionary that maps directly from block names to values"""
        return {k: v.value for k, v in self.memory.items() if v is not None}

    def list_block_names(self) -> List[str]:
        """Return a list of the block names held inside the memory object"""
        return list(self.memory.keys())

    def get_block(self, name: str) -> Block:
        """Correct way to index into the memory.memory field, returns a Block"""
        if name not in self.memory:
            return KeyError(f"Block field {name} does not exist (available sections = {', '.join(list(self.memory.keys()))})")
        else:
            return self.memory[name]

    def link_block(self, name: str, block: Block, override: Optional[bool] = False):
        """Link a new block to the memory object"""
        if not isinstance(block, Block):
            raise ValueError(f"Param block must be type Block (not {type(block)})")
        if not isinstance(name, str):
            raise ValueError(f"Name must be str (not type {type(name)})")
        if not override and name in self.memory:
            raise ValueError(f"Block with name {name} already exists")

        self.memory[name] = block

    def update_block_value(self, name: str, value: Union[List[str], str]):
        """Update the value of a block"""
        if name not in self.memory:
            raise ValueError(f"Block with name {name} does not exist")
        if not (isinstance(value, str) or (isinstance(value, list) and all(isinstance(v, str) for v in value))):
            raise ValueError(f"Provided value must be a string or list of strings")

        self.memory[name].value = value


# TODO: ideally this is refactored into ChatMemory and the subclasses are given more specific names.
class BaseChatMemory(Memory):
    def core_memory_append(self, name: str, content: str) -> Optional[str]:
        """
        Append to the contents of core memory.

        Args:
            name (str): Section of the memory to be edited (persona or human).
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        current_value = str(self.memory.get_block(name).value)
        new_value = current_value + "\n" + str(content)
        self.memory.update_block_value(name=name, value=new_value)
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
        current_value = str(self.memory.get_block(name).value)
        new_value = current_value.replace(str(old_content), str(new_content))
        self.memory.update_block_value(name=name, value=new_value)
        return None


class ChatMemory(BaseChatMemory):
    """
    ChatMemory initializes a BaseChatMemory with two default blocks
    """

    def __init__(self, persona: str, human: str, limit: int = 2000):
        super().__init__()
        self.link_block(name="persona", block=Block(name="persona", value=persona, limit=limit, label="persona"))
        self.link_block(name="human", block=Block(name="human", value=human, limit=limit, label="human"))


class BlockChatMemory(BaseChatMemory):
    """
    BlockChatMemory is a subclass of BaseChatMemory which uses shared memory blocks specified at initialization-time.
    """

    def __init__(self, blocks: List[Block] = []):
        super().__init__()
        for block in blocks:
            # TODO: centralize these internal schema validations
            assert block.name is not None and block.name != "", "each existing chat block must have a name"
            self.link_block(name=block.name, block=block)


class UpdateMemory(BaseModel):
    """Update the memory of the agent"""


class ArchivalMemorySummary(BaseModel):
    size: int = Field(..., description="Number of rows in archival memory")


class RecallMemorySummary(BaseModel):
    size: int = Field(..., description="Number of rows in recall memory")
