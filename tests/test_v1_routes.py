from unittest.mock import MagicMock, Mock, patch

import pytest
from composio.client.collections import (
    ActionModel,
    ActionParametersModel,
    ActionResponseModel,
    AppModel,
)
from fastapi.testclient import TestClient

from letta.schemas.tool import ToolCreate, ToolUpdate
from letta.server.rest_api.app import app
from letta.server.rest_api.utils import get_letta_server
from tests.helpers.utils import create_tool_from_func


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_sync_server():
    mock_server = Mock()
    app.dependency_overrides[get_letta_server] = lambda: mock_server
    return mock_server


@pytest.fixture
def add_integers_tool():
    def add(x: int, y: int) -> int:
        """
        Simple function that adds two integers.

        Parameters:
            x (int): The first integer to add.
            y (int): The second integer to add.

        Returns:
            int: The result of adding x and y.
        """
        return x + y

    tool = create_tool_from_func(add)
    yield tool


@pytest.fixture
def create_integers_tool(add_integers_tool):
    tool_create = ToolCreate(
        name=add_integers_tool.name,
        description=add_integers_tool.description,
        tags=add_integers_tool.tags,
        module=add_integers_tool.module,
        source_code=add_integers_tool.source_code,
        source_type=add_integers_tool.source_type,
        json_schema=add_integers_tool.json_schema,
    )
    yield tool_create


@pytest.fixture
def update_integers_tool(add_integers_tool):
    tool_update = ToolUpdate(
        name=add_integers_tool.name,
        description=add_integers_tool.description,
        tags=add_integers_tool.tags,
        module=add_integers_tool.module,
        source_code=add_integers_tool.source_code,
        source_type=add_integers_tool.source_type,
        json_schema=add_integers_tool.json_schema,
    )
    yield tool_update


@pytest.fixture
def composio_apps():
    affinity_app = AppModel(
        name="affinity",
        key="affinity",
        appId="3a7d2dc7-c58c-4491-be84-f64b1ff498a8",
        description="Affinity helps private capital investors to find, manage, and close more deals",
        categories=["CRM"],
        meta={
            "is_custom_app": False,
            "triggersCount": 0,
            "actionsCount": 20,
            "documentation_doc_text": None,
            "configuration_docs_text": None,
        },
        logo="https://cdn.jsdelivr.net/gh/ComposioHQ/open-logos@master/affinity.jpeg",
        docs=None,
        group=None,
        status=None,
        enabled=False,
        no_auth=False,
        auth_schemes=None,
        testConnectors=None,
        documentation_doc_text=None,
        configuration_docs_text=None,
    )
    yield [affinity_app]


@pytest.fixture
def composio_actions():
    yield [
        ActionModel(
            name="AFFINITY_GET_ALL_COMPANIES",
            display_name="Get all companies",
            parameters=ActionParametersModel(
                properties={
                    "cursor": {"default": None, "description": "Cursor for the next or previous page", "title": "Cursor", "type": "string"},
                    "limit": {"default": 100, "description": "Number of items to include in the page", "title": "Limit", "type": "integer"},
                    "ids": {"default": None, "description": "Company IDs", "items": {"type": "integer"}, "title": "Ids", "type": "array"},
                    "fieldIds": {
                        "default": None,
                        "description": "Field IDs for which to return field data",
                        "items": {"type": "string"},
                        "title": "Fieldids",
                        "type": "array",
                    },
                    "fieldTypes": {
                        "default": None,
                        "description": "Field Types for which to return field data",
                        "items": {"enum": ["enriched", "global", "relationship-intelligence"], "title": "FieldtypesEnm", "type": "string"},
                        "title": "Fieldtypes",
                        "type": "array",
                    },
                },
                title="GetAllCompaniesRequest",
                type="object",
                required=None,
            ),
            response=ActionResponseModel(
                properties={
                    "data": {"title": "Data", "type": "object"},
                    "successful": {
                        "description": "Whether or not the action execution was successful or not",
                        "title": "Successful",
                        "type": "boolean",
                    },
                    "error": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Error if any occurred during the execution of the action",
                        "title": "Error",
                    },
                },
                title="GetAllCompaniesResponse",
                type="object",
                required=["data", "successful"],
            ),
            appName="affinity",
            appId="affinity",
            tags=["companies", "important"],
            enabled=False,
            logo="https://cdn.jsdelivr.net/gh/ComposioHQ/open-logos@master/affinity.jpeg",
            description="Affinity Api Allows Paginated Access To Company Info And Custom Fields. Use `Field Ids` Or `Field Types` To Specify Data In A Request. Retrieve Field I Ds/Types Via Get `/V2/Companies/Fields`. Export Permission Needed.",
        )
    ]


# ======================================================================================================================
# Tools Routes Tests
# ======================================================================================================================
def test_delete_tool(client, mock_sync_server, add_integers_tool):
    mock_sync_server.tool_manager.delete_tool_by_id = MagicMock()

    response = client.delete(f"/v1/tools/{add_integers_tool.id}", headers={"user_id": "test_user"})

    assert response.status_code == 200
    mock_sync_server.tool_manager.delete_tool_by_id.assert_called_once_with(
        tool_id=add_integers_tool.id, actor=mock_sync_server.get_user_or_default.return_value
    )


def test_get_tool(client, mock_sync_server, add_integers_tool):
    mock_sync_server.tool_manager.get_tool_by_id.return_value = add_integers_tool

    response = client.get(f"/v1/tools/{add_integers_tool.id}", headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert response.json()["id"] == add_integers_tool.id
    assert response.json()["source_code"] == add_integers_tool.source_code
    mock_sync_server.tool_manager.get_tool_by_id.assert_called_once_with(
        tool_id=add_integers_tool.id, actor=mock_sync_server.get_user_or_default.return_value
    )


def test_get_tool_404(client, mock_sync_server, add_integers_tool):
    mock_sync_server.tool_manager.get_tool_by_id.return_value = None

    response = client.get(f"/v1/tools/{add_integers_tool.id}", headers={"user_id": "test_user"})

    assert response.status_code == 404
    assert response.json()["detail"] == f"Tool with id {add_integers_tool.id} not found."


def test_get_tool_id(client, mock_sync_server, add_integers_tool):
    mock_sync_server.tool_manager.get_tool_by_name.return_value = add_integers_tool

    response = client.get(f"/v1/tools/name/{add_integers_tool.name}", headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert response.json() == add_integers_tool.id
    mock_sync_server.tool_manager.get_tool_by_name.assert_called_once_with(
        tool_name=add_integers_tool.name, actor=mock_sync_server.get_user_or_default.return_value
    )


def test_get_tool_id_404(client, mock_sync_server):
    mock_sync_server.tool_manager.get_tool_by_name.return_value = None

    response = client.get("/v1/tools/name/UnknownTool", headers={"user_id": "test_user"})

    assert response.status_code == 404
    assert "Tool with name UnknownTool" in response.json()["detail"]


def test_list_tools(client, mock_sync_server, add_integers_tool):
    mock_sync_server.tool_manager.list_tools.return_value = [add_integers_tool]

    response = client.get("/v1/tools", headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == add_integers_tool.id
    mock_sync_server.tool_manager.list_tools.assert_called_once()


def test_create_tool(client, mock_sync_server, create_integers_tool, add_integers_tool):
    mock_sync_server.tool_manager.create_tool.return_value = add_integers_tool

    response = client.post("/v1/tools", json=create_integers_tool.model_dump(), headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert response.json()["id"] == add_integers_tool.id
    mock_sync_server.tool_manager.create_tool.assert_called_once()


def test_upsert_tool(client, mock_sync_server, create_integers_tool, add_integers_tool):
    mock_sync_server.tool_manager.create_or_update_tool.return_value = add_integers_tool

    response = client.put("/v1/tools", json=create_integers_tool.model_dump(), headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert response.json()["id"] == add_integers_tool.id
    mock_sync_server.tool_manager.create_or_update_tool.assert_called_once()


def test_update_tool(client, mock_sync_server, update_integers_tool, add_integers_tool):
    mock_sync_server.tool_manager.update_tool_by_id.return_value = add_integers_tool

    response = client.patch(f"/v1/tools/{add_integers_tool.id}", json=update_integers_tool.model_dump(), headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert response.json()["id"] == add_integers_tool.id
    mock_sync_server.tool_manager.update_tool_by_id.assert_called_once_with(
        tool_id=add_integers_tool.id, tool_update=update_integers_tool, actor=mock_sync_server.get_user_or_default.return_value
    )


def test_add_base_tools(client, mock_sync_server, add_integers_tool):
    mock_sync_server.tool_manager.add_base_tools.return_value = [add_integers_tool]

    response = client.post("/v1/tools/add-base-tools", headers={"user_id": "test_user"})

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == add_integers_tool.id
    mock_sync_server.tool_manager.add_base_tools.assert_called_once_with(actor=mock_sync_server.get_user_or_default.return_value)


def test_list_composio_apps(client, mock_sync_server, composio_apps):
    mock_sync_server.get_composio_apps.return_value = composio_apps

    response = client.get("/v1/tools/composio/apps")

    assert response.status_code == 200
    assert len(response.json()) == 1
    mock_sync_server.get_composio_apps.assert_called_once()


def test_list_composio_actions_by_app(client, mock_sync_server, composio_actions):
    mock_sync_server.get_composio_actions_from_app_name.return_value = composio_actions

    response = client.get("/v1/tools/composio/apps/App1/actions")

    assert response.status_code == 200
    assert len(response.json()) == 1
    mock_sync_server.get_composio_actions_from_app_name.assert_called_once_with(composio_app_name="App1")


def test_add_composio_tool(client, mock_sync_server, add_integers_tool):
    # Mock ToolCreate.from_composio to return the expected ToolCreate object
    with patch("letta.schemas.tool.ToolCreate.from_composio") as mock_from_composio:
        mock_from_composio.return_value = ToolCreate(
            name=add_integers_tool.name,
            source_code=add_integers_tool.source_code,
            json_schema=add_integers_tool.json_schema,
        )

        # Mock server behavior
        mock_sync_server.tool_manager.create_or_update_tool.return_value = add_integers_tool

        # Perform the request
        response = client.post(f"/v1/tools/composio/{add_integers_tool.name}", headers={"user_id": "test_user"})

        # Assertions
        assert response.status_code == 200
        assert response.json()["id"] == add_integers_tool.id
        mock_sync_server.tool_manager.create_or_update_tool.assert_called_once()

        # Verify the mocked from_composio method was called
        mock_from_composio.assert_called_once_with(action=add_integers_tool.name)
