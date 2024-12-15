from typing import List, Optional
from datetime import datetime
import numpy as np

from sqlalchemy import select, union_all, literal

from letta.constants import MAX_EMBEDDING_DIM
from letta.embeddings import embedding_model, parse_and_chunk_text
from letta.orm.errors import NoResultFound
from letta.orm.passage import AgentPassage, SourcePassage
from letta.schemas.agent import AgentState
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types



class PassageManager:
    """Manager class to handle business logic related to Passages."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def get_passage_by_id(self, passage_id: str, actor: PydanticUser) -> Optional[PydanticPassage]:
        """Fetch a passage by ID."""
        with self.session_maker() as session:
            # Try source passages first
            try:
                passage = SourcePassage.read(db_session=session, identifier=passage_id, actor=actor)
                return passage.to_pydantic()
            except NoResultFound:
                # Try archival passages
                try:
                    passage = AgentPassage.read(db_session=session, identifier=passage_id, actor=actor)
                    return passage.to_pydantic()
                except NoResultFound:
                    raise NoResultFound(f"Passage with id {passage_id} not found in database.")

    @enforce_types
    def create_passage(self, pydantic_passage: PydanticPassage, actor: PydanticUser) -> PydanticPassage:
        """Create a new passage in the appropriate table based on whether it has agent_id or source_id."""
        # Common fields for both passage types
        data = pydantic_passage.model_dump()
        common_fields = {
            "id": data.get("id"),
            "text": data["text"],
            "embedding": data["embedding"],
            "embedding_config": data["embedding_config"],
            "organization_id": data["organization_id"],
            "metadata_": data.get("metadata_", {}),
            "is_deleted": data.get("is_deleted", False),
            "created_at": data.get("created_at", datetime.utcnow()),
        }

        if "agent_id" in data and data["agent_id"]:
            assert not data.get("source_id"), "Passage cannot have both agent_id and source_id"
            agent_fields = {
                "agent_id": data["agent_id"],
            }
            passage = AgentPassage(**common_fields, **agent_fields)
        elif "source_id" in data and data["source_id"]:
            assert not data.get("agent_id"), "Passage cannot have both agent_id and source_id"
            source_fields = {
                "source_id": data["source_id"],
                "file_id": data.get("file_id"),
            }
            passage = SourcePassage(**common_fields, **source_fields)
        else:
            raise ValueError("Passage must have either agent_id or source_id")

        with self.session_maker() as session:
            passage.create(session, actor=actor)
            return passage.to_pydantic()

    @enforce_types
    def create_many_passages(self, passages: List[PydanticPassage], actor: PydanticUser) -> List[PydanticPassage]:
        """Create multiple passages."""
        return [self.create_passage(p, actor) for p in passages]

    @enforce_types
    def insert_passage(
        self,
        agent_state: AgentState,
        agent_id: str,
        text: str,
        actor: PydanticUser,
    ) -> List[PydanticPassage]:
        """Insert passage(s) into archival memory"""

        embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size
        embed_model = embedding_model(agent_state.embedding_config)

        passages = []

        try:
            # breakup string into passages
            for text in parse_and_chunk_text(text, embedding_chunk_size):
                embedding = embed_model.get_text_embedding(text)
                if isinstance(embedding, dict):
                    try:
                        embedding = embedding["data"][0]["embedding"]
                    except (KeyError, IndexError):
                        # TODO as a fallback, see if we can find any lists in the payload
                        raise TypeError(
                            f"Got back an unexpected payload from text embedding function, type={type(embedding)}, value={embedding}"
                        )
                passage = self.create_passage(
                    PydanticPassage(
                        organization_id=actor.organization_id,
                        agent_id=agent_id,
                        text=text,
                        embedding=embedding,
                        embedding_config=agent_state.embedding_config,
                    ),
                    actor=actor,
                )
                passages.append(passage)

            return passages

        except Exception as e:
            raise e

    @enforce_types
    def update_passage_by_id(self, passage_id: str, passage: PydanticPassage, actor: PydanticUser, **kwargs) -> Optional[PydanticPassage]:
        """Update a passage."""
        if not passage_id:
            raise ValueError("Passage ID must be provided.")

        with self.session_maker() as session:
            # Try source passages first
            try:
                curr_passage = SourcePassage.read(
                    db_session=session,
                    identifier=passage_id,
                    actor=actor,
                )
            except NoResultFound:
                # Try agent passages
                try:
                    curr_passage = AgentPassage.read(
                        db_session=session,
                        identifier=passage_id,
                        actor=actor,
                    )
                except NoResultFound:
                    raise ValueError(f"Passage with id {passage_id} does not exist.")

            # Update the database record with values from the provided record
            update_data = passage.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(curr_passage, key, value)

            # Commit changes
            curr_passage.update(session, actor=actor)
            return curr_passage.to_pydantic()

    @enforce_types
    def list_passages(self, 
                      actor           : PydanticUser,
                      agent_id        : Optional[str] = None, 
                      file_id         : Optional[str] = None, 
                      limit           : Optional[int] = 50,
                      query_text      : Optional[str] = None,
                      start_date      : Optional[datetime] = None,
                      end_date        : Optional[datetime] = None,
                      cursor          : Optional[str] = None,
                      source_id       : Optional[str] = None,
                      embed_query     : bool = False,
                      ascending       : bool = True,
                      embedding_config: Optional[EmbeddingConfig] = None
                     ) -> List[PydanticPassage]:
        """List passages with pagination from both source and archival passages."""        
        filters = {"organization_id": actor.organization_id}
        if file_id:
            filters["file_id"] = file_id
        if source_id:
            filters["source_id"] = source_id
        if agent_id:
            filters["agent_id"] = agent_id

        embedded_text = None
        if embed_query:
            assert embedding_config is not None
            embedded_text = embedding_model(embedding_config).get_text_embedding(query_text)
            embedded_text = np.array(embedded_text)
            embedded_text = np.pad(embedded_text, (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]), mode="constant").tolist()

        results = []

        with self.session_maker() as session:
            # Build source query with all columns
            source_query = (
                select(
                    SourcePassage,
                    literal(None).label('agent_id')  # Add agent_id as NULL
                ).where(
                    SourcePassage.organization_id == actor.organization_id
                )
            )

            # Build agent query with matching columns
            agent_query = (
                select(
                    AgentPassage.id,
                    AgentPassage.text,
                    AgentPassage.embedding_config,
                    AgentPassage.metadata_,
                    AgentPassage.embedding,
                    AgentPassage.created_at,
                    AgentPassage.updated_at,
                    AgentPassage.is_deleted,
                    AgentPassage._created_by_id,
                    AgentPassage._last_updated_by_id,
                    AgentPassage.organization_id,
                    literal(None).label('file_id'),    # Add NULL file_id
                    literal(None).label('source_id'),  # Add NULL source_id
                    AgentPassage.agent_id,
                ).where(
                    AgentPassage.organization_id == actor.organization_id
                )
            )

            # Combine queries
            combined_query = union_all(source_query, agent_query).cte('combined_passages')

            # Build main query from combined CTE
            main_query = select(combined_query)

            # Apply filters
            if start_date:
                main_query = main_query.where(combined_query.c.created_at >= start_date)
            if end_date:
                main_query = main_query.where(combined_query.c.created_at <= end_date)
            if file_id:
                main_query = main_query.where(combined_query.c.file_id == file_id)
            if source_id:
                main_query = main_query.where(combined_query.c.source_id == source_id)
            if agent_id:
                main_query = main_query.where(combined_query.c.agent_id == agent_id)

            # Vector search
            if embedded_text:
                assert query_text and embedding_config, "Vector search requires query text and embedding config"

                from letta.settings import settings
                if settings.letta_pg_uri_no_default:
                    # PostgreSQL with pgvector
                    main_query = main_query.order_by(
                        combined_query.c.embedding.cosine_distance(embedded_text).asc()
                    )
                else:
                    from sqlalchemy import func
                    # SQLite with custom vector type
                    from letta.orm.sqlite_functions import adapt_array

                    query_embedding_binary = adapt_array(embedded_text)
                    if ascending:
                        main_query = main_query.order_by(
                            func.cosine_distance(combined_query.c.embedding, query_embedding_binary).asc(),
                            combined_query.c.created_at.asc(),
                            combined_query.c.id.asc()
                        )
                    else:
                        main_query = main_query.order_by(
                            func.cosine_distance(combined_query.c.embedding, query_embedding_binary).asc(),
                            combined_query.c.created_at.desc(),
                            combined_query.c.id.asc()
                        )
            else:
                if query_text:
                    from sqlalchemy import func
                    main_query = main_query.where(func.lower(combined_query.c.text).contains(func.lower(query_text)))

            # Handle cursor-based pagination
            if cursor:
                cursor_query = select(combined_query.c.created_at).where(
                    combined_query.c.id == cursor
                ).scalar_subquery()
                
                if ascending:
                    main_query = main_query.where(
                        combined_query.c.created_at > cursor_query
                    )
                else:
                    main_query = main_query.where(
                        combined_query.c.created_at < cursor_query
                    )

            # Add ordering
            if not embed_query:  # Skip if already ordered by similarity
                if ascending:
                    main_query = main_query.order_by(
                        combined_query.c.created_at.asc(),
                        combined_query.c.id.asc(),
                    )
                else:
                    main_query = main_query.order_by(
                        combined_query.c.created_at.desc(),
                        combined_query.c.id.asc(),
                    )

            # Add limit
            if limit:
                main_query = main_query.limit(limit)

            # Execute query
            results = list(session.execute(main_query))

        passages = []
        for row in results:
            data = dict(row._mapping)
            if data['agent_id'] is not None:
                # This is an AgentPassage - remove source fields
                data.pop('source_id', None)
                data.pop('file_id', None)
                passage = AgentPassage(**data)
            else:
                # This is a SourcePassage - remove agent field
                data.pop('agent_id', None)
                passage = SourcePassage(**data)
            passages.append(passage)
        
        return [p.to_pydantic() for p in passages]

    @enforce_types
    def size(
        self,
        actor    : PydanticUser,
        agent_id : Optional[str] = None,
        source_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """Get the total count of passages with optional filters."""
        # Use list_passages to get all passages with the same filters
        passages = self.list_passages(
            actor=actor,
            agent_id=agent_id,
            source_id=source_id,
            **kwargs
        )
        return len(passages)

    @enforce_types
    def delete_passage_by_id(self, passage_id: str, actor: PydanticUser) -> bool:
        """Delete a passage from either source or archival passages."""
        if not passage_id:
            raise ValueError("Passage ID must be provided.")

        with self.session_maker() as session:
            # Try source passages first
            try:
                passage = SourcePassage.read(db_session=session, identifier=passage_id, actor=actor)
                passage.hard_delete(session, actor=actor)
                return True
            except NoResultFound:
                # Try archival passages
                try:
                    passage = AgentPassage.read(db_session=session, identifier=passage_id, actor=actor)
                    passage.hard_delete(session, actor=actor)
                    return True
                except NoResultFound:
                    raise NoResultFound(f"Passage with id {passage_id} not found.")

    def delete_passages(
        self,
        actor: PydanticUser,
        agent_id: Optional[str] = None,
        file_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        cursor: Optional[str] = None,
        query_text: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> bool:

        passages = self.list_passages(
            actor=actor,
            agent_id=agent_id,
            file_id=file_id,
            cursor=cursor,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            query_text=query_text,
            source_id=source_id,
        )

        # TODO: This is very inefficient
        # TODO: We should have a base `delete_all_matching_filters`-esque function
        for passage in passages:
            self.delete_passage_by_id(passage_id=passage.id, actor=actor)
