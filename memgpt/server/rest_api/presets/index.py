from functools import partial

from fastapi import APIRouter

from memgpt.data_types import Preset  # TODO remove
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()

"""
Implement the following functions:
* List all available presets
* Create a new preset
* Delete a preset
* TODO update a preset
"""
# TODO: delete this file


def setup_presets_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    partial(partial(get_current_user, server), password)

    return router
