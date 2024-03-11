import datetime
import requests
import uuid
from typing import Dict, List, Union, Optional, Tuple

from memgpt.data_types import AgentState, User, Preset, LLMConfig, EmbeddingConfig, Source
from memgpt.cli.cli import QuickstartChoice
from memgpt.cli.cli import set_config_with_dict, quickstart as quickstart_func, str_to_quickstart_choice
from memgpt.config import MemGPTConfig
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.metadata import MetadataStore
from memgpt.data_sources.connectors import DataConnector


def create_client(base_url: Optional[str] = None, token: Optional[str] = None):
    if base_url is None:
        return LocalClient()
    else:
        return RESTClient(base_url, token)


class AbstractClient(object):
    def __init__(
        self,
        auto_save: bool = False,
        debug: bool = False,
    ):
        self.auto_save = auto_save
        self.debug = debug

    def list_agents(self):
        """List all agents associated with a given user."""
        raise NotImplementedError

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """Check if an agent with the specified ID or name exists."""
        raise NotImplementedError

    def create_agent(
        self,
        name: Optional[str] = None,
        preset: Optional[str] = None,
        persona: Optional[str] = None,
        human: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        """Create a new agent with the specified configuration."""
        raise NotImplementedError

    def create_preset(self, preset: Preset):
        raise NotImplementedError

    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        raise NotImplementedError

    def get_agent_memory(self, agent_id: str) -> Dict:
        raise NotImplementedError

    def update_agent_core_memory(self, agent_id: str, human: Optional[str] = None, persona: Optional[str] = None) -> Dict:
        raise NotImplementedError

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        raise NotImplementedError

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def list_sources(self):
        """List loaded sources"""
        raise NotImplementedError

    def delete_source(self):
        """Delete a source and associated data (including attached to agents)"""
        raise NotImplementedError

    def load_file_into_source(self, filename: str, source_id: uuid.UUID):
        """Load {filename} and insert into source"""
        raise NotImplementedError

    def create_source(self, name: str):
        """Create a new source"""
        raise NotImplementedError

    def attach_source_to_agent(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Attach a source to an agent"""
        raise NotImplementedError

    def detach_source(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Detach a source from an agent"""
        raise NotImplementedError

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Paginated get for the archival memory for an agent"""
        raise NotImplementedError


class RESTClient(AbstractClient):
    def __init__(
        self,
        base_url: str,
        token: str,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.base_url = base_url
        self.headers = {"accept": "application/json", "authorization": f"Bearer {token}"}

    def list_agents(self):
        response = requests.get(f"{self.base_url}/agents", headers=self.headers)
        print(response.text)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        response = requests.get(f"{self.base_url}/agents/config?agent_id={str(agent_id)}", headers=self.headers)
        print(response.text)

    def create_agent(
        self,
        name: Optional[str] = None,
        preset: Optional[str] = None,
        persona: Optional[str] = None,
        human: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        if embedding_config or llm_config:
            raise ValueError("Cannot override embedding_config or llm_config when creating agent via REST API")
        payload = {
            "config": {
                "name": name,
                "preset": preset,
                "persona": persona,
                "human": human,
            }
        }
        response = requests.post(f"{self.base_url}/api/agents", json=payload, headers=self.headers)
        response_json = response.json()
        llm_config = LLMConfig(**response_json["agent_state"]["llm_config"])
        embedding_config = EmbeddingConfig(**response_json["agent_state"]["embedding_config"])
        agent_state = AgentState(
            id=uuid.UUID(response_json["agent_state"]["id"]),
            name=response_json["agent_state"]["name"],
            user_id=uuid.UUID(response_json["agent_state"]["user_id"]),
            preset=response_json["agent_state"]["preset"],
            persona=response_json["agent_state"]["persona"],
            human=response_json["agent_state"]["human"],
            llm_config=llm_config,
            embedding_config=embedding_config,
            state=response_json["agent_state"]["state"],
            # load datetime from timestampe
            created_at=datetime.datetime.fromtimestamp(response_json["agent_state"]["created_at"]),
        )
        return agent_state

    def delete_agent(self, agent_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/api/agents/{str(agent_id)}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete agent: {response.text}"

    def create_preset(self, preset: Preset):
        raise NotImplementedError

    def get_agent_config(self, agent_id: uuid.UUID) -> AgentState:
        raise NotImplementedError

    def get_agent_memory(self, agent_id: uuid.UUID) -> Dict:
        raise NotImplementedError

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
        raise NotImplementedError

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        # TODO: support role? what is return_token_count?
        payload = {"agent_id": str(agent_id), "message": message}
        response = requests.post(f"{self.base_url}/api/agents/message", json=payload, headers=self.headers)
        response_json = response.json()
        print(response_json)
        return response_json

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def list_sources(self):
        """List loaded sources"""
        response = requests.get(f"{self.base_url}/api/sources", headers=self.headers)
        response_json = response.json()
        return response_json

    def delete_source(self, source_id: uuid.UUID):
        """Delete a source and associated data (including attached to agents)"""
        response = requests.delete(f"{self.base_url}/api/sources/{str(source_id)}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete source: {response.text}"

    def load_file_into_source(self, filename: str, source_id: uuid.UUID):
        """Load {filename} and insert into source"""
        params = {"source_id": str(source_id)}
        files = {"file": open(filename, "rb")}
        response = requests.post(f"{self.base_url}/api/sources/upload", files=files, params=params, headers=self.headers)
        return response.json()

    def create_source(self, name: str) -> Source:
        """Create a new source"""
        payload = {"name": name}
        response = requests.post(f"{self.base_url}/api/sources", json=payload, headers=self.headers)
        response_json = response.json()
        print("CREATE SOURCE", response_json, response.text)
        return Source(
            id=uuid.UUID(response_json["id"]),
            name=response_json["name"],
            user_id=uuid.UUID(response_json["user_id"]),
            created_at=datetime.datetime.fromtimestamp(response_json["created_at"]),
            embedding_dim=response_json["embedding_config"]["embedding_dim"],
            embedding_model=response_json["embedding_config"]["embedding_model"],
        )

    def attach_source_to_agent(self, source_name: str, agent_id: uuid.UUID):
        """Attach a source to an agent"""
        params = {"source_name": source_name, "agent_id": agent_id}
        response = requests.post(f"{self.base_url}/api/sources/attach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to attach source to agent: {response.text}"
        return response.json()

    def detach_source(self, source_name: str, agent_id: uuid.UUID):
        """Detach a source from an agent"""
        params = {"source_name": source_name, "agent_id": str(agent_id)}
        response = requests.post(f"{self.base_url}/api/sources/detach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to detach source from agent: {response.text}"
        return response.json()

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Paginated get for the archival memory for an agent"""
        params = {"limit": limit}
        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)
        response = requests.get(f"{self.base_url}/api/agents/{str(agent_id)}/archival", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to get archival memory: {response.text}"
        return response.json()["archival_memory"]


class LocalClient(AbstractClient):
    def __init__(
        self,
        auto_save: bool = False,
        user_id: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.
        :param auto_save: indicates whether to automatically save after every message.
        :param quickstart: allows running quickstart on client init.
        :param config: optional config settings to apply after quickstart
        :param debug: indicates whether to display debug messages.
        """
        self.auto_save = auto_save

        # determine user_id (pulled from local config)
        config = MemGPTConfig.load()
        if user_id:
            self.user_id = uuid.UUID(user_id)
        else:
            self.user_id = uuid.UUID(config.anon_clientid)

        # create user if does not exist
        ms = MetadataStore(config)
        self.user = User(id=self.user_id)
        if ms.get_user(self.user_id):
            # update user
            ms.update_user(self.user)
        else:
            ms.create_user(self.user)

        # create preset records in metadata store
        from memgpt.presets.presets import add_default_presets

        add_default_presets(self.user_id, ms)

        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface=self.interface)

    def list_agents(self):
        self.interface.clear()
        return self.server.list_agents(user_id=self.user_id)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        if not (agent_id or agent_name):
            raise ValueError(f"Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError(f"Only one of agent_id or agent_name can be provided")
        existing = self.list_agents()
        if agent_id:
            return agent_id in [agent["id"] for agent in existing["agents"]]
        else:
            return agent_name in [agent["name"] for agent in existing["agents"]]

    def create_agent(
        self,
        name: Optional[str] = None,
        preset: Optional[str] = None,
        persona: Optional[str] = None,
        human: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        if name and self.agent_exists(agent_name=name):
            raise ValueError(f"Agent with name {name} already exists (user_id={self.user_id})")

        self.interface.clear()
        agent_state = self.server.create_agent(
            user_id=self.user_id,
            name=name,
            preset=preset,
            persona=persona,
            human=human,
            embedding_config=embedding_config,
            llm_config=llm_config,
        )
        return agent_state

    def create_preset(self, preset: Preset):
        preset = self.server.create_preset(preset=preset)
        return preset

    def get_agent_config(self, agent_id: str) -> AgentState:
        self.interface.clear()
        return self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id)

    def get_agent_memory(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_memory(user_id=self.user_id, agent_id=agent_id)

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
        self.interface.clear()
        return self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        self.interface.clear()
        self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        else:
            return self.interface.to_list()

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        self.interface.clear()
        return self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

    def save(self):
        self.server.save_agents()

    def load_data(self, connector: DataConnector, source_name: str):
        self.server.load_data(user_id=self.user_id, connector=connector, source_name=source_name)

    def create_source(self, name: str):
        self.server.create_source(user_id=self.user_id, name=name)

    def attach_source_to_agent(self, source_name: str, agent_id: uuid.UUID):
        self.server.attach_source_to_agent(user_id=self.user_id, source_name=source_name, agent_id=agent_id)

    def delete_agent(self, agent_id: uuid.UUID):
        self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        _, archival_json_records = self.server.get_agent_archival_cursor(
            user_id=self.user_id,
            agent_id=agent_id,
            after=after,
            before=before,
            limit=limit,
        )
        return archival_json_records
