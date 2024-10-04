from pydantic import BaseModel


class Health(BaseModel):
    """
    Health check response body
    """

    version: str
    status: str
