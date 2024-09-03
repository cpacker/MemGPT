from typing import Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from memgpt.schemas.memgpt_base import MemGPTBase

# block of the LLM context


class BaseBlock(MemGPTBase, validate_assignment=True):
    """Base block of the LLM context"""

    __id_prefix__ = "block"

    # data value
    value: Optional[str] = Field(None, description="Value of the block.")
    limit: int = Field(2000, description="Character limit of the block.")

    name: Optional[str] = Field(None, description="Name of the block.")
    template: bool = Field(False, description="Whether the block is a template (e.g. saved human/persona options).")
    label: Optional[str] = Field(None, description="Label of the block (e.g. 'human', 'persona').")

    # metadat
    description: Optional[str] = Field(None, description="Description of the block.")
    metadata_: Optional[dict] = Field({}, description="Metadata of the block.")

    # associated user/agent
    user_id: Optional[str] = Field(None, description="The unique identifier of the user associated with the block.")

    @model_validator(mode="after")
    def verify_char_limit(self) -> Self:
        try:
            assert len(self) <= self.limit
        except AssertionError:
            error_msg = f"Edit failed: Exceeds {self.limit} character limit (requested {len(self)})."
            raise ValueError(error_msg)
        except Exception as e:
            raise e
        return self

    def __len__(self):
        return len(str(self))

    def __str__(self) -> str:
        if isinstance(self.value, list):
            return ",".join(self.value)
        elif isinstance(self.value, str):
            return self.value
        else:
            return ""

    def __setattr__(self, name, value):
        """Run validation if self.value is updated"""
        super().__setattr__(name, value)
        if name == "value":
            # run validation
            self.__class__.validate(self.dict(exclude_unset=True))


class Block(BaseBlock):
    """Block of the LLM context"""

    id: str = BaseBlock.generate_id_field()
    value: str = Field(..., description="Value of the block.")


class Human(Block):
    """Human block of the LLM context"""

    label: str = "human"


class Persona(Block):
    """Persona block of the LLM context"""

    label: str = "persona"


class CreateBlock(BaseBlock):
    """Create a block"""

    template: bool = True
    label: str = Field(..., description="Label of the block.")


class CreatePersona(BaseBlock):
    """Create a persona block"""

    template: bool = True
    label: str = "persona"


class CreateHuman(BaseBlock):
    """Create a human block"""

    template: bool = True
    label: str = "human"


class UpdateBlock(BaseBlock):
    """Update a block"""

    id: str = Field(..., description="The unique identifier of the block.")
    limit: Optional[int] = Field(2000, description="Character limit of the block.")


class UpdatePersona(UpdateBlock):
    """Update a persona block"""

    label: str = "persona"


class UpdateHuman(UpdateBlock):
    """Update a human block"""

    label: str = "human"
