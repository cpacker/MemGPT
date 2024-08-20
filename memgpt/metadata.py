""" Metadata store for user/agent/data_source information"""
from typing import TYPE_CHECKING
import uuid
from typing import List, Optional
from humps import pascalize
from sqlalchemy.exc import NoResultFound
from importlib import import_module

from memgpt.log import get_logger
from memgpt.orm.utilities import get_db_session
from memgpt.orm.token import Token
from memgpt.orm.agent import Agent
from memgpt.orm.job import Job
from memgpt.orm.source import Source
from memgpt.orm.memory_templates import HumanMemoryTemplate, PersonaMemoryTemplate
from memgpt.orm.user import User
from memgpt.orm.tool import Tool

from memgpt.schemas.agent import AgentState as DataAgentState
from memgpt.orm.enums import JobStatus
from memgpt.config import MemGPTConfig
from memgpt.schemas.agent import AgentState
from memgpt.schemas.api_key import APIKey
from memgpt.schemas.block import Block, Human, Persona
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.enums import JobStatus
from memgpt.schemas.job import Job
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.memory import Memory
from memgpt.schemas.source import Source
from memgpt.schemas.tool import Tool
from memgpt.schemas.user import User
from memgpt.schemas.block import Human, Persona

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)

class MetadataStore:
    """Metadatastore acts as a bridge between the ORM and the rest of the application. Ideally it will be removed in coming PRs and
    Allow requests to handle sessions atomically (this is how FastAPI really wants things to work, and will drastically reduce the
    mucking of the ORM layer). For now, all CRUD methods are invoked here instead of the ORM layer directly.
    """
    db_session: "Session" = None

    def __init__(self, db_session: Optional["Session"] = None):
        """
        Args:
            db_session: the database session to use.
        """
        self.db_session = get_db_session()

    def create_api_key(self,
                       user_id: uuid.UUID,
                       name: Optional[str] = None,
                       actor: Optional["User"] = None) -> str:
        """Create an API key for a user
        Args:
            user_id: the user raw id as a UUID (legacy accessor)
            name: the name of the token
            actor: the user creating the API key, does not need to be the same as the user_id. will default to the user_id if not provided.
        Returns:
            api_key: the generated API key string starting with 'sk-'
        """
        token = Token(
            _user_id=actor._id or user_id,
            name=name
        ).create(self.db_session)
        return token.api_key

    def delete_api_key(self,
                       api_key: str,
                       actor: Optional["User"]=None) -> None:
        """(soft) Delete an API key from the database
        Args:
            api_key: the API key to delete
            actor: the user deleting the API key. TODO this will not be optional in the future!
        Raises:
            NotFoundError: if the API key does not exist or the user does not have access to it.
        """
        #TODO: this is a temporary shim. the long-term solution (next PR) will be to look up the token ID partial, check access, and soft delete.
        logger.info(f"User %s is deleting API key %s", actor.id, api_key)
        Token.get_by_api_key(api_key).delete(self.db_session)

    def get_api_key(self,
                    api_key: str,
                    actor: Optional["User"] = None) -> Optional[Token]:
            """legacy token lookup.
            Note: auth should remove this completely - there is no reason to look up a token without a user context.
            """
            return Token.get_by_api_key(self.db_session, api_key).to_record()

    def get_all_api_keys_for_user(self,
                                  user_id: uuid.UUID) -> List[Token]:
            """"""
            user = User.read(self.db_session, user_id)
            return [r.to_record() for r in user.tokens]

    def get_user_from_api_key(self, api_key: str) -> Optional[User]:
        """Get the user associated with a given API key"""
        return Token.get_by_api_key(self.db_session, api_key).user.to_record()

    def create_agent(self, agent: DataAgentState):
        "here is one example longhand to demonstrate the meta pattern"
        return Agent.create(self.db_session, agent.model_dump(exclude_none=True))

    def list_agents(self, user_id: uuid.UUID) -> List[DataAgentState]:
        return [a.to_record() for a in User.read(self.db_session, user_id).agents]

    def list_tools(self) -> List[DataAgentState]:
        return [a.to_record() for a in Tool.list(self.db_session)]

    def get_tool(self, name: str, user_id: uuid.UUID) -> Optional[DataAgentState]:
        return Tool.read(self.db_session, name=name).to_record()


    def __getattr__(self, name):
        """temporary metaprogramming to clean up all the getters and setters here.

        __getattr__ is always the last-ditch effort, so you can override it by declaring any method (ie `get_hamburger`) to handle the call instead.
        """
        action, raw_model_name = name.split("_",1)
        Model = getattr(import_module("memgpt.orm.__all__"), pascalize(raw_model_name).capitalize())
        if Model is None:
            raise AttributeError(f"Model {raw_model_name} action {action} not found")

        def pluralize(name):
            return name if name[-1] == "s" else name + "s"

        match action:
            case "add":
                return self.getattr("_".join(["create",raw_model_name]))
            case "get":
                # this has no support for scoping, but we won't keep this pattern long
                try:
                    def get(id, user_id = None):
                        return Model.read(self.db_session, id).to_record()
                    return get
                except IndexError:
                    raise NoResultFound(f"No {raw_model_name} found with id {id}")
            case "create":
                def create(schema):
                    splatted_pydantic = schema.model_dump(exclude_none=True)
                    return Model.create(self.db_session, splatted_pydantic).to_record()
                return create
            case "update":
                def update(schema):
                    instance = Model.read(self.db_session, schema.id)
                    splatted_pydantic = schema.model_dump(exclude_none=True, exclude=["id"])
                    for k,v in splatted_pydantic.items():
                        setattr(instance, k, v)
                    instance.update(self.db_session)
                    return instance.to_record()
                return update
            case "delete":
                def delete(*args):
                # hacky temp. look up the org for the user, get all the plural (related set) for that org and delete by name
                    if user_uuid := (args[1] if len(args) > 1 else None):
                        org = User.read(self.db_session, user_uuid).organization
                        related_set = getattr(org, pluralize(raw_model_name)) or []
                        related_set.filter(name=name).scalar().delete()
                        return
                    instance = Model.read(self.db_session, args[0])
                    instance.delete(self.db_session)
                return delete
            case "list":
                # hacky temp. look up the org for the user, get all the plural (related set) for that org
                def list(*args, **kwargs):
                    if user_uuid := kwargs.get("id"):
                        org = User.read(self.db_session, user_uuid).organization
                        return [r.to_record() for r in getattr(org, pluralize(raw_model_name)) or []]
                    # TODO: this has no scoping, no pagination, and no filtering. it's a placeholder.
                    return [r.to_record() for r in Model.list(self.db_session)]
                return list
            case _:
                raise AttributeError(f"Method {name} not found")

    def update_human(self, human: Human) -> "Human":
        sql_human = HumanMemoryTemplate(**human.model_dump(exclude_none=True)).create(self.db_session)
        return sql_human.to_record()

    def update_persona(self, persona: Persona) -> "Persona":
        sql_persona = PersonaMemoryTemplate(**persona.model_dump(exclude_none=True)).create(self.db_session)
        return sql_persona.to_record()

    def get_all_users(self, cursor: Optional[uuid.UUID] = None, limit: Optional[int] = 50) -> (Optional[uuid.UUID], List[User]):
        del limit # TODO: implement pagination as part of predicate
        return None , [u.to_record() for u in User.list(self.db_session)]

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
        org = User.read(self.db_session, user_id)
        return org.human_memory_templates.filter(name=name).scalar()

    def get_persona(self, name: str, user_id: uuid.UUID) -> Optional[Persona]:
        org = User.read(self.db_session, user_id)
        return org.human_memory_templates.filter(name=name).scalar()

    def update_job_status(self, job_id: uuid.UUID, status: JobStatus):
        job = Job.read(self.db_session, job_id)
        job.status = status
        job.update(self.db_session)
