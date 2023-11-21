from fastapi import APIRouter

from memgpt import utils
from memgpt.config import AgentConfig
from pydantic import BaseModel


router = APIRouter()


class AgentPostDTO(BaseModel):
    name: str


@router.get("/agents", tags=["agents"])
async def available_agents():
    agent_files = utils.list_agent_config_files()
    agents = [AgentConfig.load(f).__dict__ for f in agent_files]
    return agents


@router.post("/agents", tags=["agents"])
async def create_agent(agent: AgentPostDTO):
    agent_config = AgentConfig(
        name=agent.name,
        human="cs_phd",
        persona="sam_pov",
    )

    # save new agent config
    agent_config.save()

    return agent_config.__dict__
