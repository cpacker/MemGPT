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

        with self.session_maker() as session:# return list(session.execute(query).scalars())
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

    def _get_excluded_roles(self, include_system_messages: bool = False, include_tool_messages: bool = False) -> List[str]:
        """Helper method to get list of roles to exclude based on include flags.
        
        Args:
            include_system_messages: If True, don't exclude system messages
            include_tool_messages: If True, don't exclude tool messages
            
        Returns:
            List of role names to exclude
        """
        excluded_roles = []
        if not include_system_messages:
            excluded_roles.append("system")
        if not include_tool_messages:
            excluded_roles.append("tool")
        return excluded_roles

    @enforce_types
    def size(
        self, 
        actor: PydanticUser, 
        filters: Optional[Dict] = None,
        only_user_messages: bool = True,
    ) -> int:
        """Get the total count of messages with optional filters.
        
        Args:
            actor: The user requesting the count
            filters: Additional filters to apply
            only_user_messages: If True, only count user messages
        """
        with self.session_maker() as session:
            query = session.query(MessageModel).filter(MessageModel.user_id == actor.id)
            
            # Handle role exclusions
            if only_user_messages:
                query = query.filter(MessageModel.role == "user")

            # Add any additional filters
            if filters:
                for key, value in filters.items():
                    query = query.filter(getattr(MessageModel, key) == value)
            
            # import ipdb;ipdb.set_trace()
            return query.count()

    @enforce_types
    def list_messages(
        self,
        actor: PydanticUser,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
        only_user_messages: bool = True,
    ) -> List[PydanticMessage]:
        """List messages with flexible filtering and pagination options.
        
        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            filters: Additional filters to apply
            query_text: Optional text to search for in message content
            only_user_messages: If True, include only user messages in results
            
        Returns:
            List[PydanticMessage] - List of messages matching the criteria
        """
        with self.session_maker() as session:
            # Start with base filters
            message_filters = {"user_id": actor.id}
            if only_user_messages:
                message_filters["role"] = "user"
            
            # Add any additional filters
            if filters:
                message_filters.update(filters)

            results = MessageModel.list(
                db_session=session,
                cursor=cursor,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                query_text=query_text,
                **message_filters
            )

            return [msg.to_pydantic() for msg in results]
