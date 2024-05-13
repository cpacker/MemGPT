import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemGPTCredentials:
    # openai config
    openai_key: Optional[str] = None

    @classmethod
    def load(cls) -> "MemGPTCredentials":
        opeani_key = os.getenv("OPENAI_API_KEY")
        assert opeani_key, "OPENAI_API_KEY environment variable must be set"
        return cls(openai_key=opeani_key)
