from typing import List, Optional
from letta.constants import MAX_EMBEDDING_DIM
from datetime import datetime
import numpy as np

from letta.orm.errors import NoResultFound
from letta.orm.passage import AgentPassage, SourcePassage
from letta.utils import enforce_types

from letta.embeddings import embedding_model, parse_and_chunk_text
from letta.schemas.embedding_config import EmbeddingConfig

from letta.schemas.agent import AgentState
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
    def insert_passage(self, 
        agent_state: AgentState,
        agent_id: str,
        text: str, 
        actor: PydanticUser, 
    ) -> List[PydanticPassage]:
        """ Insert passage(s) into archival memory """

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
                        embedding_config=agent_state.embedding_config
                    ),
                    actor=actor
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
                      source_id       : Optional[str] = None,
                      embed_query     : bool = False,
                      ascending       : bool = True,
                      embedding_config: Optional[EmbeddingConfig] = None
                     ) -> List[PydanticPassage]:
        """List passages with pagination from both source and archival passages."""        
        filters = {"organization_id": actor.organization_id}
        if file_id:
            filters["file_id"] = file_id
        
        embedded_text = None
        if embed_query:
            assert embedding_config is not None
            embedded_text = embedding_model(embedding_config).get_text_embedding(query_text)
            embedded_text = np.array(embedded_text)
            embedded_text = np.pad(embedded_text, (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]), mode="constant").tolist()

        results = []

        with self.session_maker() as session:

            # Query source passages if source_id is specified or no specific table filter is given
            if source_id or (not agent_id and not source_id): # could be querying all passages in an organization
                source_filters = {**filters}
                if source_id:
                    source_filters["source_id"] = source_id
                
                try:
                    source_results = SourcePassage.list(
                        db_session=session,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit,
                        query_text=query_text if not embedded_text else None,
                        query_embedding=embedded_text,
                        ascending=ascending,
                        **source_filters
                    )
                    results.extend(source_results)
                except NoResultFound:
                    pass

            # Query archival passages if agent_id is specified or no specific table filter is given
            if agent_id or (not agent_id and not source_id): # could be querying all passages in an organization
                archival_filters = {**filters}
                if agent_id:
                    archival_filters["agent_id"] = agent_id
                
                try:
                    archival_results = AgentPassage.list(
                        db_session=session,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit,
                        query_text=query_text if not embedded_text else None,
                        query_embedding=embedded_text,
                        ascending=ascending,
                        **archival_filters
                    )
                    results.extend(archival_results)
                except NoResultFound:
                    pass

        # Sort combined results by similarity or created_at and apply limit
        if embed_query:
            # Convert query embedding to numpy array for efficient computation
            query_embedding = np.array(embedded_text)
            
            # NOTE: this might be slow but it's less messy than modifying Base.list() to handle the Passages edge case
            # Calculate cosine similarity for each passage
            def get_distance(passage):
                passage_embedding = np.array(passage.embedding)
                similarity = np.dot(query_embedding, passage_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(passage_embedding)
                )
                return 1 - similarity
            
            results.sort(key=lambda x: (get_distance(x), x.created_at))
        else:
            results.sort(key=lambda x: x.created_at, reverse=not ascending)

        if limit:
            results = results[:limit]
        
        return [p.to_pydantic() for p in results]

    @enforce_types
    def size(
        self,
        actor    : PydanticUser,
        agent_id : Optional[str] = None,
        source_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """Get the total count of passages with optional filters."""
        with self.session_maker() as session:
            total = 0
            
            # Count source passages if source_id is specified or no specific filter is given
            if source_id or (not agent_id and not source_id):
                source_filters = {**kwargs}
                if source_id:
                    source_filters["source_id"] = source_id
                total += SourcePassage.size(db_session=session, actor=actor, **source_filters)
            
            # Count archival passages if agent_id is specified or no specific filter is given
            if agent_id or (not agent_id and not source_id):
                archival_filters = {**kwargs}
                if agent_id:
                    archival_filters["agent_id"] = agent_id
                total += AgentPassage.size(db_session=session, actor=actor, **archival_filters)
            
            return total

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

    def delete_passages(self,
                        actor: PydanticUser,
                        agent_id: Optional[str] = None,
                        file_id: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: Optional[int] = 50,
                        cursor: Optional[str] = None,
                        query_text: Optional[str] = None,
                        source_id: Optional[str] = None
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
            source_id=source_id)
        
        for passage in passages:
            self.delete_passage_by_id(passage_id=passage.id, actor=actor)
