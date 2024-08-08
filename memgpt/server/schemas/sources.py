from typing import Optional, List
from fastapi import UploadFile
from pydantic import BaseModel, Field


from memgpt.models.pydantic_models import DocumentModel, PassageModel, SourceModel

class ListSourcesResponse(BaseModel):
    sources: List[SourceModel] = Field(..., description="List of available sources.")


class CreateSourceRequest(BaseModel):
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")


class UploadFileToSourceRequest(BaseModel):
    file: UploadFile = Field(..., description="The file to upload.")


class UploadFileToSourceResponse(BaseModel):
    source: SourceModel = Field(..., description="The source the file was uploaded to.")
    added_passages: int = Field(..., description="The number of passages added to the source.")
    added_documents: int = Field(..., description="The number of documents added to the source.")


class GetSourcePassagesResponse(BaseModel):
    passages: List[PassageModel] = Field(..., description="List of passages from the source.")


class GetSourceDocumentsResponse(BaseModel):
    documents: List[DocumentModel] = Field(..., description="List of documents from the source.")
