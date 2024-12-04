from typing import List, Optional, Dict, Tuple
from datetime import datetime
from letta.orm.errors import NoResultFound
from letta.utils import enforce_types

from letta.orm.message import Message as MessageModel
from letta.schemas.message import Message as PydanticMessage

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
    def update_message_by_id(self, message_id: str, message: PydanticMessage) -> Optional[PydanticMessage]:
        """
        Updates an existing record in the database with values from the provided record object.

        Replaces db.py:SQLLiteStorageConnector.update() 
        """
        if not message_id:
            raise ValueError("Message ID must be provided.")

        with self.session_maker() as session:
            try:
                # Fetch existing message from database
                msg = MessageModel.read(db_session=session, identifier=message_id)
                if not msg:
                    raise ValueError(f"Message with id {message_id} does not exist.")

                # Update the database record with values from the provided record
                for column in MessageModel.__table__.columns:
                    column_name = column.name
                    if hasattr(message, column_name):
                        new_value = getattr(message, column_name)
                        setattr(msg, column_name, new_value)

                # Commit changes
                msg.update(session)
                return msg.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def delete_message_by_id(self, message_id: str) -> bool:
        """Delete a message."""
        with self.session_maker() as session:
            try:
                msg = MessageModel.read(
                    db_session=session, 
                    identifier=message_id
                )
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
            results = MessageModel.list(
                db_session=session, 
                cursor=cursor,
                limit=limit,
                filters=filters
            )
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
            results = MessageModel.list(
                db_session=session, 
                limit=limit, 
                cursor=cursor, 
                filters=filters
            )
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def get_all(self, cursor: Optional[str] = None, limit: Optional[int] = None) -> List[PydanticMessage]:
        """Get all messages with optional pagination."""
        with self.session_maker() as session:
            query = session.query(MessageModel)
            
            if start is not None:
                query = query.offset(start)
            if count is not None:
                query = query.limit(count)
                
            results = query.all()
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def size(self, filters: Optional[Dict] = None) -> int:
        """Get the total count of messages with optional filters."""
        with self.session_maker() as session:
            query = session.query(MessageModel)
            
            if filters:
                for key, value in filters.items():
                    query = query.filter(
                        getattr(MessageModel, key) == value
                    )
            
            return query.count()
