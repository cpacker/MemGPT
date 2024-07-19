import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from memgpt.schemas.openai.chat_completions import ToolCall


class Tool(BaseModel):

    tool_id: uuid.UUID = Field(..., description="The unique identifier of the tool.")
    name: str = Field(..., description="The name of the tool.")
    description: Optional[str] = Field(None, description="The description of the tool.")

    name: str = Field(..., description="The name of the function.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the function.", primary_key=True)
    tags: List[str] = Field(..., description="Metadata tags.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    source_code: Optional[str] = Field(..., description="The source code of the function.")
    module: Optional[str] = Field(None, description="The module of the function.")

    json_schema: Dict = Field(default_factory=dict, description="The JSON schema of the function.")

    # optional: user_id (user-specific tools)
    user_id: Optional[uuid.UUID] = Field(None, description="The unique identifier of the user associated with the function.")

    def to_dict(self):
        """Convert into OpenAI representation"""
        return vars(
            ToolCall(
                tool_id=self.id,
                tool_call_type="function",
                function=self.module,
            )
        )
