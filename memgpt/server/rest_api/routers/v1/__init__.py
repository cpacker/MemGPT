from memgpt.server.rest_api.routers.v1.tools import router as tools_router
from memgpt.server.rest_api.routers.v1.sources import router as sources_router
from memgpt.server.rest_api.routers.v1.presets import router as presets_router
from memgpt.server.rest_api.routers.v1.agents import router as agents_router
from memgpt.server.rest_api.routers.v1.large_language_models import router as llm_router
from memgpt.server.rest_api.routers.v1.personas import router as personas_router

ROUTERS = [
    tools_router,
    sources_router,
    presets_router,
    agents_router,
    llm_router,
    personas_router,
]