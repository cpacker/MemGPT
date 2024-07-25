from pydantic import BaseModel


class MemGPTUsageStatistics(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    step_count: int
