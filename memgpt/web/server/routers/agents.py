from fastapi import APIRouter

from memgpt import utils
from memgpt.config import AgentConfig
from memgpt.connectors.storage import StorageConnector

router = APIRouter()


@router.get("/agents", tags=["agents"])
async def available_agents():
    agent_files = utils.list_agent_config_files()
    agents = [AgentConfig.load(f).__dict__ for f in agent_files]
    return agents
