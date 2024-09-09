from typing import List, Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field

from memgpt.schemas.document import Document
from memgpt.schemas.passage import Passage
from memgpt.schemas.source import Source


class ListSourcesResponse(BaseModel):
    sources: List[Source] = Field(..., description="List of available sources.")


class CreateSourceRequest(BaseModel):
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")


class UploadFileToSourceRequest(BaseModel):
    file: UploadFile = Field(..., description="The file to upload.")


class UploadFileToSourceResponse(BaseModel):
    source: Source = Field(..., description="The source the file was uploaded to.")
    added_passages: int = Field(..., description="The number of passages added to the source.")
    added_documents: int = Field(..., description="The number of documents added to the source.")


class GetSourcePassagesResponse(BaseModel):
    passages: List[Passage] = Field(..., description="List of passages from the source.")


class GetSourceDocumentsResponse(BaseModel):
    documents: List[Document] = Field(..., description="List of documents from the source.")
