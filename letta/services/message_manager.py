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

    def create_many_messages(self, messages: List[PydanticMessage]) -> List[PydanticMessage]:
        """Create multiple messages."""

        # Can't use enforce_types here, so manually check
        assert all(isinstance(msg, PydanticMessage) for msg in messages)

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
                        **filters
                      )
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def query_date_range(self, user_id: str, start_date: datetime, end_date: datetime,
                        agent_id: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticMessage]:
        """Query messages within a date range."""
        with self.session_maker() as session:
            # If start_date equals end_date, add a small buffer to include messages created at that exact time
            if start_date.date() == end_date.date():
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            query = (
                session.query(MessageModel)
                .filter(MessageModel.user_id == user_id)
                .filter(MessageModel.created_at >= start_date)
                .filter(MessageModel.created_at <= end_date)
                .filter(MessageModel.role != "system")
                .filter(MessageModel.role != "tool")
            )
            
            if agent_id:
                query = query.filter(MessageModel.agent_id == agent_id)
                
            if limit:
                query = query.limit(limit)
                
            results = query.all()
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def query_text(self, user_id: str, query_text: str, agent_id: Optional[str] = None,
                  limit: Optional[int] = 50) -> List[PydanticMessage]:
        """Search messages by text content."""
        from sqlalchemy import func
        
        with self.session_maker() as session:
            query = (
                session.query(MessageModel)
                .filter(MessageModel.user_id == user_id)
                .filter(func.lower(MessageModel.text).contains(func.lower(query_text)))
                .filter(MessageModel.role != "system")
                .filter(MessageModel.role != "tool")
            )
            
            if agent_id:
                query = query.filter(MessageModel.agent_id == agent_id)
                
            if limit:
                query = query.limit(limit)
                
            results = query.all()
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def get_all(self, cursor: Optional[str] = None, limit: Optional[int] = None) -> List[PydanticMessage]:
        """Get all messages with optional pagination."""
        with self.session_maker() as session:
            query = session.query(MessageModel)
            
            if cursor is not None:
                query = query.offset(cursor)
            if limit is not None:
                query = query.limit(limit)
                
            results = query.all()
            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def get_all_cursor(
        self,
        filters: Optional[Dict] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Optional[int] = 1000,
        order_by: str = "created_at",
        reverse: bool = False,
    ) -> Tuple[Optional[str], List[PydanticMessage]]:
        """Get all messages with cursor-based pagination.

        Args:
            filters: Optional dictionary of filters to apply
            after: Return records after this cursor (exclusive)
            before: Return records before this cursor (exclusive)
            limit: Maximum number of records to return
            order_by: Field to sort by
            reverse: If True, sort in descending order

        Returns:
            Tuple of (next_cursor, list of messages)
        """
        from sqlalchemy import desc, asc, or_, and_

        with self.session_maker() as session:
            query = session.query(MessageModel)

            # Apply filters if provided
            if filters:
                for field, value in filters.items():
                    query = query.filter(getattr(MessageModel, field) == value)

            # Sort by the specified field and ID as tiebreaker
            if reverse:
                query = query.order_by(desc(getattr(MessageModel, order_by)), asc(MessageModel.id))
            else:
                query = query.order_by(asc(getattr(MessageModel, order_by)), asc(MessageModel.id))

            # Handle cursor-based pagination
            if after:
                after_msg = self.get_message_by_id(after)
                if after_msg:
                    after_value = getattr(after_msg, order_by)
                    sort_exp = getattr(MessageModel, order_by) > after_value
                    query = query.filter(
                        or_(
                            sort_exp,
                            and_(
                                getattr(MessageModel, order_by) == after_value,
                                MessageModel.id > after
                            )
                        )
                    )

            if before:
                before_msg = self.get_message_by_id(before)
                if before_msg:
                    before_value = getattr(before_msg, order_by)
                    sort_exp = getattr(MessageModel, order_by) < before_value
                    query = query.filter(
                        or_(
                            sort_exp,
                            and_(
                                getattr(MessageModel, order_by) == before_value,
                                MessageModel.id < before
                            )
                        )
                    )

            # Get records with limit
            if limit:
                query = query.limit(limit)

            results = query.all()

            if not results:
                return None, []

            messages = [msg.to_pydantic() for msg in results]
            next_cursor = results[-1].id

            return next_cursor, messages

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
