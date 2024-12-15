from typing import List, Optional

from letta.orm.errors import NoResultFound
from letta.orm.file import FileMetadata as FileMetadataModel
from letta.orm.source import Source as SourceModel
from letta.schemas.agent import AgentState as PydanticAgentState
from letta.schemas.file import FileMetadata as PydanticFileMetadata
from letta.schemas.source import Source as PydanticSource
from letta.schemas.source import SourceUpdate
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types, printd


class SourceManager:
    """Manager class to handle business logic related to Sources."""

    def __init__(self):
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_source(self, source: PydanticSource, actor: PydanticUser) -> PydanticSource:
        """Create a new source based on the PydanticSource schema."""
        # Try getting the source first by id
        db_source = self.get_source_by_id(source.id, actor=actor)
        if db_source:
            return db_source
        else:
            with self.session_maker() as session:
                # Provide default embedding config if not given
                source.organization_id = actor.organization_id
                source = SourceModel(**source.model_dump(exclude_none=True))
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
            source.hard_delete(db_session=session, actor=actor)
            return source.to_pydantic()

    @enforce_types
    def list_sources(self, actor: PydanticUser, cursor: Optional[str] = None, limit: Optional[int] = 50, **kwargs) -> List[PydanticSource]:
        """List all sources with optional pagination."""
        with self.session_maker() as session:
            sources = SourceModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                organization_id=actor.organization_id,
                **kwargs,
            )
            return [source.to_pydantic() for source in sources]

    @enforce_types
    def list_attached_agents(self, source_id: str, actor: Optional[PydanticUser] = None) -> List[PydanticAgentState]:
        """
        Lists all agents that have the specified source attached.

        Args:
            source_id: ID of the source to find attached agents for
            actor: User performing the action (optional for now, following existing pattern)

        Returns:
            List[PydanticAgentState]: List of agents that have this source attached
        """
        with self.session_maker() as session:
            # Verify source exists and user has permission to access it
            source = SourceModel.read(db_session=session, identifier=source_id, actor=actor)

            # The agents relationship is already loaded due to lazy="selectin" in the Source model
            # and will be properly filtered by organization_id due to the OrganizationMixin
            return [agent.to_pydantic() for agent in source.agents]

    # TODO: We make actor optional for now, but should most likely be enforced due to security reasons
    @enforce_types
    def get_source_by_id(self, source_id: str, actor: Optional[PydanticUser] = None) -> Optional[PydanticSource]:
        """Retrieve a source by its ID."""
        with self.session_maker() as session:
            try:
                source = SourceModel.read(db_session=session, identifier=source_id, actor=actor)
                return source.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def get_source_by_name(self, source_name: str, actor: PydanticUser) -> Optional[PydanticSource]:
        """Retrieve a source by its name."""
        with self.session_maker() as session:
            sources = SourceModel.list(
                db_session=session,
                name=source_name,
                organization_id=actor.organization_id,
                limit=1,
            )
            if not sources:
                return None
            else:
                return sources[0].to_pydantic()

    @enforce_types
    def create_file(self, file_metadata: PydanticFileMetadata, actor: PydanticUser) -> PydanticFileMetadata:
        """Create a new file based on the PydanticFileMetadata schema."""
        db_file = self.get_file_by_id(file_metadata.id, actor=actor)
        if db_file:
            return db_file
        else:
            with self.session_maker() as session:
                file_metadata.organization_id = actor.organization_id
                file_metadata = FileMetadataModel(**file_metadata.model_dump(exclude_none=True))
                file_metadata.create(session, actor=actor)
            return file_metadata.to_pydantic()

    # TODO: We make actor optional for now, but should most likely be enforced due to security reasons
    @enforce_types
    def get_file_by_id(self, file_id: str, actor: Optional[PydanticUser] = None) -> Optional[PydanticFileMetadata]:
        """Retrieve a file by its ID."""
        with self.session_maker() as session:
            try:
                file = FileMetadataModel.read(db_session=session, identifier=file_id, actor=actor)
                return file.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def list_files(
        self, source_id: str, actor: PydanticUser, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[PydanticFileMetadata]:
        """List all files with optional pagination."""
        with self.session_maker() as session:
            files = FileMetadataModel.list(
                db_session=session, cursor=cursor, limit=limit, organization_id=actor.organization_id, source_id=source_id
            )
            return [file.to_pydantic() for file in files]

    @enforce_types
    def delete_file(self, file_id: str, actor: PydanticUser) -> PydanticFileMetadata:
        """Delete a file by its ID."""
        with self.session_maker() as session:
            file = FileMetadataModel.read(db_session=session, identifier=file_id)
            file.hard_delete(db_session=session, actor=actor)
            return file.to_pydantic()
