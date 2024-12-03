from typing import List, Optional, Dict, Tuple
from datetime import datetime
from letta.orm.errors import NoResultFound
from letta.utils import enforce_types

from letta.orm.message import Message as MessageModel, Passage as PassageModel
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.passage import Passage as PydanticPassage

class MessageManager:
    """Manager class to handle business logic related to Messages."""

    def __init__(self):
        from letta.server.server import db_context
        self.session_maker = db_context

    @enforce_types
    def get_message_by_id(self, message_id: str) -> Optional[PydanticMessage]:
        """Fetch a message by ID."""
        with self.session_maker() as session:
            try:
                message = MessageModel.read(db_session=session, identifier=message_id)
                return message.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def create_message(self, pydantic_msg: PydanticMessage) -> PydanticMessage:
        """Create a new message."""
        with self.session_maker() as session:
            msg = MessageModel(**pydantic_msg.model_dump())
            msg.create(session)
            return msg.to_pydantic()

    @enforce_types
    def create_many_messages(self, messages: List[PydanticMessage]) -> List[PydanticMessage]:
        """Create multiple messages."""
        with self.session_maker() as session:
            msg_models = [MessageModel(**msg.model_dump()) for msg in messages]
            for msg in msg_models:
                msg.create(session)
            return [msg.to_pydantic() for msg in msg_models]

    @enforce_types
    def update_message(self, message_id: str, **kwargs) -> Optional[PydanticMessage]:
        """Update a message."""
        with self.session_maker() as session:
            try:
                msg = MessageModel.read(db_session=session, identifier=message_id)
                for key, value in kwargs.items():
                    setattr(msg, key, value)
                msg.update(session)
                return msg.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def delete_message(self, message_id: str) -> bool:
        """Delete a message."""
        with self.session_maker() as session:
            try:
                msg = MessageModel.read(db_session=session, identifier=message_id)
                msg.hard_delete(session)
                return True
            except NoResultFound:
                return False

    @enforce_types
    def list_messages(self, user_id: str, agent_id: Optional[str] = None,
                     cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticMessage]:
        """List messages with pagination based on cursor and limit."""
        with self.session_maker() as session:
            filters = {"user_id": user_id}
            if agent_id:
                filters["agent_id"] = agent_id
            results = MessageModel.list(db_session=session, cursor=cursor, limit=limit, filters=filters)
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def query_date_range(self, user_id: str, start_date: datetime, end_date: datetime,
                        agent_id: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticMessage]:
        """Query messages within a date range."""
        with self.session_maker() as session:
            filters = {
                "user_id": user_id,
                "created_at_gte": start_date,
                "created_at_lte": end_date
            }
            if agent_id:
                filters["agent_id"] = agent_id
            results = MessageModel.list(db_session=session, limit=limit, filters=filters)
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def query_text(self, user_id: str, query_text: str, agent_id: Optional[str] = None,
                  limit: Optional[int] = 50) -> List[PydanticMessage]:
        """Search messages by text content."""
        with self.session_maker() as session:
            filters = {
                "user_id": user_id,
                "text_contains": query_text.lower()
            }
            if agent_id:
                filters["agent_id"] = agent_id
            results = MessageModel.list(db_session=session, limit=limit, filters=filters)
            return [msg.to_pydantic() for msg in results]


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