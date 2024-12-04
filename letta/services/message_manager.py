from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime
from letta.orm.errors import NoResultFound
from letta.utils import enforce_types

from letta.orm.message import Message as MessageModel
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.user import User as PydanticUser

class MessageManager:
    """Manager class to handle business logic related to Messages."""

    def __init__(self):
        from letta.server.server import db_context
        self.session_maker = db_context

    @enforce_types
    def get_message_by_id(self, message_id: str, actor: PydanticUser) -> Optional[PydanticMessage]:
        """Fetch a message by ID."""
        with self.session_maker() as session:
            try:
                message = MessageModel.read(
                    db_session=session, 
                    identifier=message_id, 
                    actor=actor, 
                    access_type="user"
                )
                return message.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def create_message(self, pydantic_msg: PydanticMessage, actor: PydanticUser) -> PydanticMessage:
        """Create a new message."""
        with self.session_maker() as session:
            pydantic_msg.user_id = actor.id
            msg_data = pydantic_msg.model_dump()
            msg = MessageModel(**msg_data)
            msg.create(session, actor=actor) # Persist to database
            return msg.to_pydantic()

    def create_many_messages(self, messages: List[PydanticMessage], actor: PydanticUser) -> List[PydanticMessage]:
        """Create multiple messages."""

        # Can't use enforce_types here, so manually check
        assert all(isinstance(msg, PydanticMessage) for msg in messages)

        with self.session_maker() as session:
            # Set user_id for each message
            for msg in messages:
                msg.user_id = actor.id
            msg_models = [MessageModel(**msg.model_dump()) for msg in messages]
            for msg in msg_models:
                msg.create(session, actor=actor)
            return [msg.to_pydantic() for msg in msg_models]

    @enforce_types
    def update_message_by_id(self, message_id: str, message: PydanticMessage, actor: PydanticUser) -> Optional[PydanticMessage]:
        """
        Updates an existing record in the database with values from the provided record object.

        Replaces db.py:SQLLiteStorageConnector.update() 
        """
        if not message_id:
            raise ValueError("Message ID must be provided.")

        with self.session_maker() as session:
            try:
                # Fetch existing message from database
                msg = MessageModel.read(
                    db_session=session, 
                    identifier=message_id, 
                    actor=actor,
                    access_type="user",
                )
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
    def delete_message_by_id(self, message_id: str, actor: PydanticUser) -> bool:
        """Delete a message."""
        with self.session_maker() as session:
            try:
                msg = MessageModel.read(
                    db_session=session, 
                    identifier=message_id,
                    actor=actor,
                    access_type="user",
                )
                msg.delete(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Message with id {message_id} not found.")

    @enforce_types
    def size(self, actor: PydanticUser, filters: Optional[Dict] = None) -> int:
        """Get the total count of messages with optional filters."""
        with self.session_maker() as session:
            query = session.query(MessageModel).filter(MessageModel.user_id == actor.id)
            
            if filters:
                for key, value in filters.items():
                    query = query.filter(
                        getattr(MessageModel, key) == value
                    )
            
            return query.count()

    @enforce_types
    def list_messages(
        self,
        actor: PydanticUser,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        reverse: bool = False,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
    ) -> List[PydanticMessage]:
        """List messages with flexible filtering and pagination options.
        
        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            reverse: If True, sort in descending order
            filters: Additional filters to apply
            query_text: Optional text to search for in message content
            
        Returns:
            List[PydanticMessage] - List of messages matching the criteria
        """
        with self.session_maker() as session:
            results = MessageModel.list(
                db_session=session,
                cursor=cursor,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                reverse=reverse,
                query_text=query_text,
                user_id=actor.id,
                **(filters or {})
            )

            return [msg.to_pydantic() for msg in results]
