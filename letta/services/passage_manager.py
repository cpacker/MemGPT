from typing import List, Optional, Dict, Tuple
from datetime import datetime
from letta.orm.errors import NoResultFound
from letta.utils import enforce_types

from letta.schemas.passage import Passage as PydanticPassage

class PassageManager:
    """Manager class to handle business logic related to Passages."""

    def __init__(self):
        from letta.server.server import db_context
        self.session_maker = db_context

    @enforce_types
    def get_passage_by_id(self, passage_id: str) -> Optional[PydanticPassage]:
        """Fetch a passage by ID."""
        with self.session_maker() as session:
            try:
                passage = PassageModel.read(db_session=session, identifier=passage_id)
                return passage.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def create_passage(self, pydantic_passage: PydanticPassage) -> PydanticPassage:
        """Create a new passage."""
        with self.session_maker() as session:
            passage = PassageModel(**pydantic_passage.model_dump())
            passage.create(session)
            return passage.to_pydantic()

    @enforce_types
    def create_many_passages(self, passages: List[PydanticPassage]) -> List[PydanticPassage]:
        """Create multiple passages."""
        with self.session_maker() as session:
            passage_models = [PassageModel(**p.model_dump()) for p in passages]
            for passage in passage_models:
                passage.create(session)
            return [p.to_pydantic() for p in passage_models]

    @enforce_types
    def update_passage(self, passage_id: str, **kwargs) -> Optional[PydanticPassage]:
        """Update a passage."""
        with self.session_maker() as session:
            try:
                passage = PassageModel.read(db_session=session, identifier=passage_id)
                for key, value in kwargs.items():
                    setattr(passage, key, value)
                passage.update(session)
                return passage.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def delete_passage(self, passage_id: str) -> bool:
        """Delete a passage."""
        with self.session_maker() as session:
            try:
                passage = PassageModel.read(db_session=session, identifier=passage_id)
                passage.hard_delete(session)
                return True
            except NoResultFound:
                return False

    @enforce_types
    def list_passages(self, user_id: str, agent_id: Optional[str] = None, 
                     file_id: Optional[str] = None, cursor: Optional[str] = None, 
                     limit: Optional[int] = 50) -> List[PydanticPassage]:
        """List passages with pagination."""
        with self.session_maker() as session:
            filters = {"user_id": user_id}
            if agent_id:
                filters["agent_id"] = agent_id
            if file_id:
                filters["file_id"] = file_id
            results = PassageModel.list(db_session=session, cursor=cursor, limit=limit, filters=filters)
            return [p.to_pydantic() for p in results]