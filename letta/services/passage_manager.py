from typing import List, Optional, Dict, Tuple
from datetime import datetime
from letta.orm.errors import NoResultFound
from letta.utils import enforce_types

from letta.orm.passage import Passage as PassageModel
from letta.orm.sqlalchemy_base import AccessType
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.user import User as PydanticUser

class PassageManager:
    """Manager class to handle business logic related to Passages."""

    def __init__(self):
        from letta.server.server import db_context
        self.session_maker = db_context

    @enforce_types
    def get_passage_by_id(self, passage_id: str, actor: PydanticUser) -> Optional[PydanticPassage]:
        """Fetch a passage by ID."""
        with self.session_maker() as session:
            try:
                passage = PassageModel.read(db_session=session, identifier=passage_id, actor=actor, access_type=AccessType.USER)
                return passage.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def create_passage(self, pydantic_passage: PydanticPassage, actor: PydanticUser) -> PydanticPassage:
        """Create a new passage."""
        with self.session_maker() as session:
            passage = PassageModel(**pydantic_passage.model_dump())
            passage.create(session, actor=actor)
            return passage.to_pydantic()

    @enforce_types
    def create_many_passages(self, passages: List[PydanticPassage], actor: PydanticUser) -> List[PydanticPassage]:
        """Create multiple passages."""
        return [self.create_passage(p, actor) for p in passages]

    @enforce_types
    def update_passage_by_id(self, passage_id: str, passage: PydanticPassage, actor: PydanticUser, **kwargs) -> Optional[PydanticPassage]:
        """Update a passage."""
        if not passage_id:
            raise ValueError("Passage ID must be provided.")

        with self.session_maker() as session:
            try:
                # Fetch existing message from database
                curr_passage = PassageModel.read(
                    db_session=session,
                    identifier=passage_id,
                    actor=actor,
                    access_type=AccessType.USER,
                )
                if not curr_passage:
                    raise ValueError(f"Passage with id {passage_id} does not exist.")

                # Update the database record with values from the provided record
                update_data = passage.model_dump(exclude_unset=True, exclude_none=True)
                for key, value in update_data.items():
                    setattr(curr_passage, key, value)

                # Commit changes
                curr_passage.update(session, actor=actor)
                return curr_passage.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def delete_passage_by_id(self, passage_id: str, actor: PydanticUser) -> bool:
        """Delete a passage."""
        if not passage_id:
            raise ValueError("Passage ID must be provided.")

        with self.session_maker() as session:
            try:
                passage = PassageModel.read(db_session=session, identifier=passage_id, actor=actor, access_type=AccessType.USER)
                passage.hard_delete(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Passage with id {passage_id} not found.")

    @enforce_types
    def list_passages(self, 
                      actor     : PydanticUser,
                      agent_id  : Optional[str] = None, 
                      file_id   : Optional[str] = None, 
                      cursor    : Optional[str] = None, 
                      limit     : Optional[int] = 50,
                      query_text: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date  : Optional[datetime] = None
                     ) -> List[PydanticPassage]:
        """List passages with pagination."""
        with self.session_maker() as session:
            filters = {"user_id": actor.id}
            if agent_id:
                filters["agent_id"] = agent_id
            if file_id:
                filters["file_id"] = file_id
            if query_text:
                filters["query_text"] = query_text
            if start_date:
                filters["start_date"] = start_date
            if end_date:
                filters["end_date"] = end_date
            results = PassageModel.list(db_session=session, cursor=cursor, limit=limit, **filters)
            return [p.to_pydantic() for p in results]
    
    @enforce_types
    def size(
        self,
        actor    : PydanticUser,
        agent_id : Optional[str] = None,
    ) -> int:
        """Get the total count of messages with optional filters.

        Args:
            actor   : The user requesting the count
            agent_id: The agent ID
        """
        with self.session_maker() as session:
            return PassageModel.size(db_session=session, actor=actor, agent_id=agent_id, access_type=AccessType.USER)

    @enforce_types
    def list_passages_for_agent(
        self,
        actor: PydanticUser,
        agent_id: Optional[str] = None,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
    ) -> List[PydanticPassage]:
        """List messages with flexible filtering and pagination options.

        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            filters: Additional filters to apply
            query_text: Optional text to search for in message content

        Returns:
            List[PydanticPassage] - List of messages matching the criteria
        """
        with self.session_maker() as session:
            # Start with base filters
            passage_filters = {}
            if agent_id:
                passage_filters.update({"agent_id": agent_id})
            if actor:
                passage_filters.update({"organization_id": actor.organization_id})
            if filters:
                passage_filters.update(filters)

            results = PassageModel.list(
                db_session=session,
                cursor=cursor,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                query_text=query_text,
                **passage_filters,
            )

            return [passage.to_pydantic() for passage in results]