from datetime import datetime

from pydantic import BaseModel


class User(BaseModel):
    user_id: str
    name: str
    created_at: datetime
