from datetime import datetime
from typing import List, Optional

import numpy as np

from letta.constants import MAX_EMBEDDING_DIM
from letta.embeddings import embedding_model, parse_and_chunk_text
from letta.orm.errors import NoResultFound
from letta.orm.passage import Passage as PassageModel
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
            passage = PassageModel.read(db_session=session, identifier=passage_id, actor=actor)
            return passage.to_pydantic()

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
            # Fetch existing message from database
            curr_passage = PassageModel.read(
                db_session=session,
                identifier=passage_id,
                actor=actor,
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

    @enforce_types
    def delete_passage_by_id(self, passage_id: str, actor: PydanticUser) -> bool:
        """Delete a passage."""
        if not passage_id:
            raise ValueError("Passage ID must be provided.")

        with self.session_maker() as session:
            try:
                passage = PassageModel.read(db_session=session, identifier=passage_id, actor=actor)
                passage.hard_delete(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Passage with id {passage_id} not found.")

    @enforce_types
    def list_passages(
        self,
        actor: PydanticUser,
        agent_id: Optional[str] = None,
        file_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ascending: bool = True,
        source_id: Optional[str] = None,
        embed_query: bool = False,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> List[PydanticPassage]:
        """List passages with pagination."""
        with self.session_maker() as session:
            filters = {"organization_id": actor.organization_id}
            if agent_id:
                filters["agent_id"] = agent_id
            if file_id:
                filters["file_id"] = file_id
            if source_id:
                filters["source_id"] = source_id

            embedded_text = None
            if embed_query:
                assert embedding_config is not None

                # Embed the text
                embedded_text = embedding_model(embedding_config).get_text_embedding(query_text)

                # Pad the embedding with zeros
                embedded_text = np.array(embedded_text)
                embedded_text = np.pad(embedded_text, (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]), mode="constant").tolist()

            results = PassageModel.list(
                db_session=session,
                cursor=cursor,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                ascending=ascending,
                query_text=query_text if not embedded_text else None,
                query_embedding=embedded_text,
                **filters,
            )
            return [p.to_pydantic() for p in results]

    @enforce_types
    def size(self, actor: PydanticUser, agent_id: Optional[str] = None, **kwargs) -> int:
        """Get the total count of messages with optional filters.

        Args:
            actor   : The user requesting the count
            agent_id: The agent ID
        """
        with self.session_maker() as session:
            return PassageModel.size(db_session=session, actor=actor, agent_id=agent_id, **kwargs)

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
