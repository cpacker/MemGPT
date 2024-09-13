from typing import TYPE_CHECKING, Dict, List, Optional

from jinja2 import Template, TemplateSyntaxError
from pydantic import BaseModel, Field

# Forward referencing to avoid circular import with Agent -> Memory -> Agent
if TYPE_CHECKING:
    from memgpt.agent import Agent

from memgpt.schemas.block import Block


class Memory(BaseModel, validate_assignment=True):
    """

    Represents the in-context memory of the agent. This includes both the `Block` objects (labelled by sections), as well as tools to edit the blocks.

    Attributes:
        memory (Dict[str, Block]): Mapping from memory block section to memory block.

    """

    # Memory.memory is a dict mapping from memory block section to memory block.
    memory: Dict[str, Block] = Field(default_factory=dict, description="Mapping from memory block section to memory block.")

    # Memory.template is a Jinja2 template for compiling memory module into a prompt string.
    prompt_template: str = Field(
        default="{% for block in memory.values() %}"
        '<{{ block.name }} characters="{{ block.value|length }}/{{ block.limit }}">\n'
        "{{ block.value }}\n"
        "</{{ block.name }}>"
        "{% if not loop.last %}\n{% endif %}"
        "{% endfor %}",
        description="Jinja2 template for compiling memory blocks into a prompt string",
    )

    def get_prompt_template(self) -> str:
        """Return the current Jinja2 template string."""
        return str(self.prompt_template)

    def set_prompt_template(self, prompt_template: str):
        """
        Set a new Jinja2 template string.
        Validates the template syntax and compatibility with current memory structure.
        """
        try:
            # Validate Jinja2 syntax
            Template(prompt_template)

            # Validate compatibility with current memory structure
            test_render = Template(prompt_template).render(memory=self.memory)

            # If we get here, the template is valid and compatible
            self.prompt_template = prompt_template
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 template syntax: {str(e)}")
        except Exception as e:
            raise ValueError(f"Prompt template is not compatible with current memory structure: {str(e)}")

    @classmethod
    def load(cls, state: dict):
        """Load memory from dictionary object"""
        obj = cls()
        if len(state.keys()) == 2 and "memory" in state and "prompt_template" in state:
            # New format
            obj.prompt_template = state["prompt_template"]
            for key, value in state["memory"].items():
                obj.memory[key] = Block(**value)
        else:
            # Old format (pre-template)
            for key, value in state.items():
                obj.memory[key] = Block(**value)
        return obj

    def compile(self) -> str:
        """Generate a string representation of the memory in-context using the Jinja2 template"""
        template = Template(self.prompt_template)
        return template.render(memory=self.memory)

    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "memory": {key: value.model_dump() for key, value in self.memory.items()},
            "prompt_template": self.prompt_template,
        }

    def to_flat_dict(self):
        """Convert to a dictionary that maps directly from block names to values"""
        return {k: v.value for k, v in self.memory.items() if v is not None}

    def list_block_names(self) -> List[str]:
        """Return a list of the block names held inside the memory object"""
        return list(self.memory.keys())

    def get_block(self, name: str) -> Block:
        """Correct way to index into the memory.memory field, returns a Block"""
        if name not in self.memory:
            raise KeyError(f"Block field {name} does not exist (available sections = {', '.join(list(self.memory.keys()))})")
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

    def update_block_value(self, name: str, value: str):
        """Update the value of a block"""
        if name not in self.memory:
            raise ValueError(f"Block with name {name} does not exist")
        if not isinstance(value, str):
            raise ValueError(f"Provided value must be a string")

        self.memory[name].value = value


# TODO: ideally this is refactored into ChatMemory and the subclasses are given more specific names.
class BasicBlockMemory(Memory):
    """
    BasicBlockMemory is a basic implemention of the Memory class, which takes in a list of blocks and links them to the memory object. These are editable by the agent via the core memory functions.

    Attributes:
        memory (Dict[str, Block]): Mapping from memory block section to memory block.

    Methods:
        core_memory_append: Append to the contents of core memory.
        core_memory_replace: Replace the contents of core memory.
    """

    def __init__(self, blocks: List[Block] = []):
        """
        Initialize the BasicBlockMemory object with a list of pre-defined blocks.

        Args:
            blocks (List[Block]): List of blocks to be linked to the memory object.
        """
        super().__init__()
        for block in blocks:
            # TODO: centralize these internal schema validations
            assert block.name is not None and block.name != "", "each existing chat block must have a name"
            self.link_block(name=block.name, block=block)

    def core_memory_append(self: "Agent", name: str, content: str) -> Optional[str]:  # type: ignore
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

    def core_memory_replace(self: "Agent", name: str, old_content: str, new_content: str) -> Optional[str]:  # type: ignore
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


class ChatMemory(BasicBlockMemory):
    """
    ChatMemory initializes a BaseChatMemory with two default blocks, `human` and `persona`.
    """

    def __init__(self, persona: str, human: str, limit: int = 2000):
        """
        Initialize the ChatMemory object with a persona and human string.

        Args:
            persona (str): The starter value for the persona block.
            human (str): The starter value for the human block.
            limit (int): The character limit for each block.
        """
        super().__init__()
        self.link_block(name="persona", block=Block(name="persona", value=persona, limit=limit, label="persona"))
        self.link_block(name="human", block=Block(name="human", value=human, limit=limit, label="human"))


class UpdateMemory(BaseModel):
    """Update the memory of the agent"""


class ArchivalMemorySummary(BaseModel):
    size: int = Field(..., description="Number of rows in archival memory")


class RecallMemorySummary(BaseModel):
    size: int = Field(..., description="Number of rows in recall memory")


class CreateArchivalMemory(BaseModel):
    text: str = Field(..., description="Text to write to archival memory.")
