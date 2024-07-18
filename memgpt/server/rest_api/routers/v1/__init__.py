from memgpt.server.rest_api.routers.v1.tools import tools_router
from memgpt.server.rest_api.routers.v1.sources import sources_router
from memgpt.server.rest_api.routers.v1.presets import presets_router

ROUTERS = [
    tools_router,
    sources_router,
    presets_router,
]