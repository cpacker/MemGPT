from typing import List, Optional

from letta.orm.errors import NoResultFound
from letta.orm.source import Source as SourceModel
from letta.schemas.source import Source as PydanticSource
from letta.schemas.source import SourceCreate, SourceUpdate
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types, printd


class SourceManager:
    """Manager class to handle business logic related to Sources."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_source(self, source_create: SourceCreate, actor: PydanticUser) -> PydanticSource:
        """Create a new source based on the SourceCreate schema."""
        with self.session_maker() as session:
            # Provide default embedding config if not given
            create_data = source_create.model_dump()
            source = SourceModel(**create_data, organization_id=actor.organization_id)
            source.create(session, actor=actor)
        return source.to_pydantic()

    @enforce_types
    def update_source(self, source_id: str, source_update: SourceUpdate, actor: PydanticUser) -> PydanticSource:
        """Update a source by its ID with the given SourceUpdate object."""
        with self.session_maker() as session:
            source = SourceModel.read(db_session=session, identifier=source_id, actor=actor)

            # get update dictionary
            update_data = source_update.model_dump(exclude_unset=True, exclude_none=True)
            # Remove redundant update fields
            update_data = {key: value for key, value in update_data.items() if getattr(source, key) != value}

            if update_data:
                for key, value in update_data.items():
                    setattr(source, key, value)
                source.update(db_session=session, actor=actor)
            else:
                printd(
                    f"`update_source` was called with user_id={actor.id}, organization_id={actor.organization_id}, name={source.name}, but found existing source with nothing to update."
                )

            return source.to_pydantic()

    @enforce_types
    def delete_source(self, source_id: str, actor: PydanticUser) -> PydanticSource:
        """Delete a source by its ID."""
        with self.session_maker() as session:
            source = SourceModel.read(db_session=session, identifier=source_id)
            source.delete(db_session=session, actor=actor)
            return source.to_pydantic()

    @enforce_types
    def list_sources(self, actor: PydanticUser, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticSource]:
        """List all sources with optional pagination."""
        with self.session_maker() as session:
            sources = SourceModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                organization_id=actor.organization_id,
            )
            return [source.to_pydantic() for source in sources]

    # TODO: We make actor optional for now, but should most likely be enforced due to security reasons
    @enforce_types
    def get_source_by_id(self, source_id: str, actor: Optional[PydanticUser] = None) -> PydanticSource:
        """Retrieve a source by its ID."""
        with self.session_maker() as session:
            source = SourceModel.read(db_session=session, identifier=source_id, actor=actor)
            return source.to_pydantic()

    @enforce_types
    def get_source_by_name(self, source_name: str, actor: PydanticUser) -> PydanticSource:
        """Retrieve a source by its name."""
        with self.session_maker() as session:
            sources = SourceModel.list(
                db_session=session,
                name=source_name,
                organization_id=actor.organization_id,
                limit=1,
            )
            if not sources:
                raise NoResultFound(f"Source with name '{source_name}' not found.")
            return sources[0].to_pydantic()
