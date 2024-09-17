""" Metadata store for user/agent/data_source information"""

import uuid
from importlib import import_module
from typing import TYPE_CHECKING, List, Optional, Tuple

from humps import pascalize
from sqlalchemy.exc import NoResultFound

from memgpt.log import get_logger
from memgpt.orm.agent import Agent
from memgpt.orm.enums import JobStatus
from memgpt.orm.errors import NoResultFound
from memgpt.orm.job import Job
from memgpt.orm.memory_templates import HumanMemoryTemplate, PersonaMemoryTemplate
from memgpt.orm.message import Message as SQLMessage
from memgpt.orm.organization import Organization as SQLOrganization
from memgpt.orm.source import Source
from memgpt.orm.token import Token
from memgpt.orm.tool import Tool as SQLTool
from memgpt.orm.user import User as SQLUser
from memgpt.orm.utilities import get_db_session
from memgpt.schemas.agent import AgentState as DataAgentState
from memgpt.schemas.block import Block, Human, Persona
from memgpt.schemas.enums import JobStatus
from memgpt.schemas.job import Job
from memgpt.schemas.message import Message
from memgpt.schemas.source import Source
from memgpt.schemas.tool import Tool
from memgpt.schemas.user import User

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)


class MetadataStore:
    """Metadatastore acts as a bridge between the ORM and the rest of the application. Ideally it will be removed in coming PRs and
    Allow requests to handle sessions atomically (this is how FastAPI really wants things to work, and will drastically reduce the
    mucking of the ORM layer). For now, all CRUD methods are invoked here instead of the ORM layer directly.
    """

    db_session: "Session" = None

    def __init__(self, db_session: Optional["Session"] = None, actor: Optional["User"] = None):
        """
        Args:
            db_session: the database session to use.
            actor: the user making the request. should be a straight pass from server.get_current_user at the moment
                and when we collapse metadatastore into server this will no longer be necessary.
        """
        self.db_session = db_session or get_db_session()
        self.actor = actor

    def create_api_key(self, user_id: uuid.UUID, name: Optional[str] = None, actor: Optional["User"] = None) -> str:
        """Create an API key for a user
        Args:
            user_id: the user raw id as a UUID (legacy accessor)
            name: the name of the token
            actor: the user creating the API key, does not need to be the same as the user_id. will default to the user_id if not provided.
        Returns:
            api_key: the generated API key string starting with 'sk-'
        """
        token = Token(_user_id=actor.id or user_id, name=name).create(self.db_session)
        return token.api_key

    def delete_api_key(self, api_key: str) -> None:
        """(soft) Delete an API key from the database
        Args:
            api_key: the API key to delete
        Raises:
            NotFoundError: if the API key does not exist or the user does not have access to it.
        """
        # TODO: this is a temporary shim. the long-term solution (next PR) will be to look up the token ID partial, check access, and soft delete.
        logger.info(f"User %s is deleting API key %s", self.actor.id, api_key)
        Token.get_by_api_key(api_key).delete(self.db_session)

    def get_api_key(self, api_key: str, actor: Optional["User"] = None) -> Optional[Token]:
        """legacy token lookup.
        Note: auth should remove this completely - there is no reason to look up a token without a user context.
        """
        return Token.get_by_api_key(self.db_session, api_key).to_pydantic()

    def get_all_api_keys_for_user(self, user_id: uuid.UUID) -> List[Token]:
        """"""
        user = SQLUser.read(self.db_session, user_id)
        return [r.to_pydantic() for r in user.tokens]

    def get_user_from_api_key(self, api_key: str) -> Optional[User]:
        """Get the user associated with a given API key"""
        return Token.get_by_api_key(self.db_session, api_key).user.to_pydantic()

    def _clean_agent_state(self, agent_state: DataAgentState, action: str = "create") -> DataAgentState:
        """Clean an agent state before creating or updating it in DB"""
        excluded_fields = [
            "user_id",
            "memory",
            "created_at",
            "tools",
            "message_ids",
            "messages",
        ]
        if action == "create":
            excluded_fields.append("id")

        splatted_pydantic = agent_state.model_dump(exclude_none=True, exclude=excluded_fields)

        actor = SQLUser.read(db_session=self.db_session, identifier=SQLUser.to_uid(self.actor.id))
        
        if agent_state.memory:
            splatted_pydantic["core_memory"] = []
            for name, block in agent_state.memory.memory.items():
                block.organization_id = actor.organization_id
                splatted_pydantic["core_memory"].append(block.to_sqlalchemy(self.db_session))

        if agent_state.tools:
            # tools need to be read by name
            splatted_pydantic["tools"] = [
                SQLTool.read(self.db_session, name=r, actor=actor) if not isinstance(r, Tool) else r.to_sqlalchemy(self.db_session)
                for r in agent_state.tools
            ]
        if agent_state.message_ids:
            splatted_pydantic["messages"] = [
                SQLMessage.read(self.db_session, identifier=r, actor=actor) if not isinstance(r, Message) else r.to_sqlalchemy()
                for r in agent_state.message_ids
            ]

        # Blocks/Memory are a bit more complex, so we'll handle them separately in controller

        return splatted_pydantic

    def create_agent(self, agent_state: DataAgentState):
        """Create an agent from a DataAgentState
        *Note* There is not currently a clear SQL <> Pydantic mapping for this object.
        Args:
            agent: the agent to create"""
        return (
            Agent(created_by_id=self.actor.id, **self._clean_agent_state(agent_state=agent_state, action="create"))
            .create(self.db_session)
            .to_pydantic()
        )

    def update_agent(self, agent_state: DataAgentState):
        """Create an agent from a DataAgentState
        *Note* There is not currently a clear SQL <> Pydantic mapping for this object.
        Args:
            agent: the agent to create"""
        instance = Agent.read(self.db_session, agent_state.id)
        splatted_pydantic = self._clean_agent_state(agent_state=agent_state, action="update")
        for k, v in splatted_pydantic.items():
            setattr(instance, k, v)
        instance.update(self.db_session)

        return instance.to_pydantic()

    def list_agents(self, **kwargs) -> List[DataAgentState]:
        return self.list_agent(**kwargs)

    def list_tools(self, **kwargs) -> List[Tool]:
        return self.list_tool(**kwargs)

    def list_blocks(self, **kwargs) -> List[Block]:
        return self.list_block(**kwargs)

    def get_organization(self, name: str = "Default Organization") -> SQLOrganization:
        return SQLOrganization.default(self.db_session)

    def get_tool(self, id: Optional[str] = None, name: Optional[str] = None, user_id: Optional[str] = None) -> "Tool":
        actor = self._get_actor(user_id)
        if id:
            return SQLTool.read(db_session=self.db_session, actor=actor, identifier=id).to_pydantic()
        if name:
            return SQLTool.read(db_session=self.db_session, actor=actor, name=name).to_pydantic()

    def _get_actor(self, user_id: Optional[str] = None) -> "SQLUser":
        if user_id:
            return SQLUser.read(db_session=self.db_session, identifier=SQLUser.to_uid(user_id))
        else:
            return self.actor.to_sqlalchemy(self.db_session)

    def __getattr__(self, name):
        """temporary metaprogramming to clean up all the getters and setters here.

        __getattr__ is always the last-ditch effort, so you can override it by declaring any method (ie `get_hamburger`) to handle the call instead.
        """
        action, raw_model_name = name.split("_", 1)
        Model = getattr(import_module("memgpt.orm.__all__"), pascalize(raw_model_name).capitalize())
        if Model is None:
            raise AttributeError(f"Model {raw_model_name} action {action} not found")

        def pluralize(name):
            return name if name[-1] == "s" else name + "s"

        match action:
            case "add":
                return self.getattr("_".join(["create", raw_model_name]))
            case "get":
                # this has no support for scoping, but we won't keep this pattern long
                try:
                    def get(id, user_id=None):
                        return Model.read(db_session=self.db_session, identifier=id, actor=self._get_actor(user_id)).to_pydantic()

                    return get
                except IndexError:
                    raise NoResultFound(f"No {raw_model_name} found with id {id}")
            case "create":

                def create(schema):
                    splatted_pydantic = schema.model_dump(exclude_none=True)
                    splatted_pydantic = {k: v for k, v in splatted_pydantic.items() if hasattr(Model, k)}

                    return Model(created_by_id=self.actor.id, **splatted_pydantic).create(self.db_session).to_pydantic()

                return create
            case "update":

                def update(schema):
                    instance = Model.read(db_session=self.db_session, identifier=schema.id, actor=self._get_actor())
                    splatted_pydantic = schema.model_dump(exclude_none=True, exclude=["id"])
                    for k, v in splatted_pydantic.items():
                        setattr(instance, k, v)
                    instance.update(self.db_session)
                    return instance.to_pydantic()

                return update
            case "delete":

                def delete(*args):
                    # hacky temp. look up the org for the user, get all the plural (related set) for that org and delete by name
                    if user_uuid := (args[1] if len(args) > 1 else None):
                        org = SQLUser.read(self.db_session, user_uuid).organization
                        related_set = getattr(org, pluralize(raw_model_name)) or []
                        related_set.filter(name=name).scalar().delete()
                        return
                    instance = Model.read(self.db_session, args[0])
                    instance.delete(self.db_session)

                return delete
            case "list":
                # hacky temp. look up the org for the user, get all the plural (related set) for that org
                def list(*args, **kwargs):
                    filters = kwargs.get("filters", {})
                    if user_uuid := kwargs.get("user_id"):
                        filters["_organization_id"] = SQLUser.read(self.db_session, user_uuid).organization._id
                    return [r.to_pydantic() for r in Model.list(self.db_session, self._get_actor(), **filters)]

                return list
            case _:
                raise AttributeError(f"Method {name} not found")

    def update_human(self, human: Human) -> "Human":
        sql_human = HumanMemoryTemplate(**human.model_dump(exclude_none=True)).create(self.db_session)
        return sql_human.to_pydantic()

    def update_persona(self, persona: Persona) -> "Persona":
        sql_persona = PersonaMemoryTemplate(**persona.model_dump(exclude_none=True)).create(self.db_session)
        return sql_persona.to_pydantic()

    # def update_block(self, block: Block) -> Block:
    #    # TODO: change this to be general, not hard coded to human/persona
    #    if block.label == "human":
    #        self.update_human(block)
    #    elif block.label == "persona":
    #        self.update_persona(block)
    #    else:
    #        raise ValueError(f"Block label {block.label} not recognized")

    def get_all_users(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> Tuple[Optional[str], List[User]]:
        del limit  # TODO: implement pagination as part of predicate
        return None, self.list_user(user_id=self.actor.id)

    # agent source metadata
    def attach_source(self, user_id: uuid.UUID, agent_id: uuid.UUID, source_id: uuid.UUID) -> None:
        agent = Agent.read(self.db_session, agent_id)
        source = Source.read(self.db_session, source_id)
        agent.sources.append(source)

    def list_attached_sources(self, agent_id: uuid.UUID) -> List[uuid.UUID]:
        return [s._id for s in Agent.read(self.db_session, agent_id).sources]

    def list_attached_agents(self, source_id: uuid.UUID) -> List[uuid.UUID]:
        return [a._id for a in Source.read(self.db_session, source_id).agents]

    def detach_source(self, agent_id: uuid.UUID, source_id: uuid.UUID) -> None:
        agent = Agent.read(self.db_session, agent_id)
        source = Source.read(self.db_session, source_id)
        agent.sources.remove(source)

    def get_human(self, name: str, user_id: uuid.UUID) -> Optional[Human]:
        org = SQLUser.read(self.db_session, user_id)
        return org.human_memory_templates.filter(name=name).scalar()

    def get_persona(self, name: str, user_id: uuid.UUID) -> Optional[Persona]:
        org = SQLUser.read(self.db_session, user_id)
        return org.human_memory_templates.filter(name=name).scalar()

    def update_job_status(self, job_id: uuid.UUID, status: JobStatus):
        job = Job.read(self.db_session, job_id)
        job.status = status
        job.update(self.db_session)
