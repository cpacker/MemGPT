from typing import List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

# block of the LLM context


class Block(BaseModel, validate_assignment=True):
    """Block of the LLM context"""

    description: Optional[str] = Field(None, description="Description of the block.")
    limit: int = Field(2000, description="Character limit of the block.")
    value: Optional[Union[List[str], str]] = Field(None, description="Value of the block.")

    def __setattr__(self, name, value):
        """Run validation if self.value is updated"""
        super().__setattr__(name, value)
        if name == "value":
            # run validation
            self.__class__.validate(self.dict(exclude_unset=True))

    @model_validator(mode="after")
    def verify_char_limit(self) -> Self:
        if len(self) >= self.limit:
            error_msg = f"Edit failed: Exceeds {self.limit} character limit (requested {len(self)})."
            raise ValueError(error_msg)
        return self

    # @validator("value", always=True)
    # def check_value_length(cls, v, values):
    #    if v is not None:
    #        # Fetching the limit from the values dictionary
    #        limit = values.get("limit", 2000)  # Default to 2000 if limit is not yet set

    #        # Check if the value exceeds the limit
    #        if isinstance(v, str):
    #            length = len(v)
    #        elif isinstance(v, list):
    #            length = sum(len(item) for item in v)
    #        else:
    #            raise ValueError("Value must be either a string or a list of strings.")

    #        if length > limit:
    #            error_msg = f"Edit failed: Exceeds {limit} character limit (requested {length})."
    #            # TODO: add archival memory error?
    #            raise ValueError(error_msg)
    #    return v

    def __len__(self):
        return len(str(self))

    def __str__(self) -> str:
        if isinstance(self.value, list):
            return ",".join(self.value)
        elif isinstance(self.value, str):
            return self.value
        else:
            return ""
