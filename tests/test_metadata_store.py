import pytest

from memgpt.agent import Agent, save_agent
from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
from memgpt.data_types import AgentState, LLMConfig, Source, User
from memgpt.metadata import MetadataStore
from memgpt.models.pydantic_models import HumanModel, PersonaModel
from memgpt.presets.presets import add_default_presets
from memgpt.settings import settings
from memgpt.utils import get_human_text, get_persona_text
from tests import TEST_MEMGPT_CONFIG


# @pytest.mark.parametrize("storage_connector", ["postgres", "sqlite"])
@pytest.mark.parametrize("storage_connector", ["sqlite"])
def test_storage(storage_connector):
    if storage_connector == "postgres":
        TEST_MEMGPT_CONFIG.archival_storage_uri = settings.pg_uri
        TEST_MEMGPT_CONFIG.recall_storage_uri = settings.pg_uri
        TEST_MEMGPT_CONFIG.archival_storage_type = "postgres"
        TEST_MEMGPT_CONFIG.recall_storage_type = "postgres"
    if storage_connector == "sqlite":
        TEST_MEMGPT_CONFIG.recall_storage_type = "local"

    ms = MetadataStore(TEST_MEMGPT_CONFIG)

    # users
    user_1 = User()
    user_2 = User()
    ms.create_user(user_1)
    ms.create_user(user_2)

    # test adding default humans/personas/presets
    # add_default_humans_and_personas(user_id=user_1.id, ms=ms)
    # add_default_humans_and_personas(user_id=user_2.id, ms=ms)
    ms.add_human(human=HumanModel(name="test_human", text="This is a test human"))
    ms.add_persona(persona=PersonaModel(name="test_persona", text="This is a test persona"))
    add_default_presets(user_id=user_1.id, ms=ms)
    add_default_presets(user_id=user_2.id, ms=ms)
    assert len(ms.list_humans(user_id=user_1.id)) > 0, ms.list_humans(user_id=user_1.id)
    assert len(ms.list_personas(user_id=user_1.id)) > 0, ms.list_personas(user_id=user_1.id)

    # generate data
    agent_1 = AgentState(
        user_id=user_1.id,
        name="agent_1",
        preset=DEFAULT_PRESET,
        persona=DEFAULT_PERSONA,
        human=DEFAULT_HUMAN,
        llm_config=TEST_MEMGPT_CONFIG.default_llm_config,
        embedding_config=TEST_MEMGPT_CONFIG.default_embedding_config,
    )
    source_1 = Source(user_id=user_1.id, name="source_1")

    # test creation
    ms.create_agent(agent_1)
    ms.create_source(source_1)

    # test listing
    len(ms.list_agents(user_id=user_1.id)) == 1
    len(ms.list_agents(user_id=user_2.id)) == 0
    len(ms.list_sources(user_id=user_1.id)) == 1
    len(ms.list_sources(user_id=user_2.id)) == 0

    # test agent_state saving
    agent_state = ms.get_agent(agent_1.id).state
    assert agent_state == {}, agent_state  # when created via create_agent, it should be empty

    from memgpt.presets.presets import add_default_presets

    add_default_presets(user_1.id, ms)
    preset_obj = ms.get_preset(name=DEFAULT_PRESET, user_id=user_1.id)
    from memgpt.interface import CLIInterface as interface  # for printing to terminal

    # Overwrite fields in the preset if they were specified
    preset_obj.human = get_human_text(DEFAULT_HUMAN)
    preset_obj.persona = get_persona_text(DEFAULT_PERSONA)

    # Create the agent
    agent = Agent(
        interface=interface(),
        created_by=user_1.id,
        name="agent_test_agent_state",
        preset=preset_obj,
        llm_config=config.default_llm_config,
        embedding_config=config.default_embedding_config,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=(
            True if (config.default_llm_config.model is not None and "gpt-4" in config.default_llm_config.model) else False
        ),
    )
    agent_with_agent_state = agent.agent_state
    save_agent(agent=agent, ms=ms)

    agent_state = ms.get_agent(agent_with_agent_state.id).state
    assert agent_state is not None, agent_state  # when created via create_agent_from_preset, it should be non-empty

    # test: updating

    # test: update JSON-stored LLMConfig class
    print(agent_1.llm_config, TEST_MEMGPT_CONFIG.default_llm_config)
    llm_config = ms.get_agent(agent_1.id).llm_config
    assert isinstance(llm_config, LLMConfig), f"LLMConfig is {type(llm_config)}"
    assert llm_config.model == "gpt-4", f"LLMConfig model is {llm_config.model}"
    llm_config.model = "gpt3.5-turbo"
    agent_1.llm_config = llm_config
    ms.update_agent(agent_1)
    assert ms.get_agent(agent_1.id).llm_config.model == "gpt3.5-turbo", f"Updated LLMConfig to {ms.get_agent(agent_1.id).llm_config.model}"

    # test attaching sources
    len(ms.list_attached_sources(agent_id=agent_1.id)) == 0
    ms.attach_source(user_1.id, agent_1.id, source_1.id)
    len(ms.list_attached_sources(agent_id=agent_1.id)) == 1

    # test: detaching sources
    ms.detach_source(agent_1.id, source_1.id)
    len(ms.list_attached_sources(agent_id=agent_1.id)) == 0

    # test getting
    ms.get_user(user_1.id)
    ms.get_agent(agent_1.id)
    ms.get_source(source_1.id)

    # test api keys
    api_key = ms.create_api_key(user_id=user_1.id)
    print("api_key=", api_key.token, api_key.user_id)
    api_key_result = ms.get_api_key(api_key=api_key.token)
    assert api_key.token == api_key_result.token, (api_key, api_key_result)
    user_result = ms.get_user_from_api_key(api_key=api_key.token)
    assert user_1.id == user_result.id, (user_1, user_result)
    all_keys_for_user = ms.get_all_api_keys_for_user(user_id=user_1.id)
    assert len(all_keys_for_user) > 0, all_keys_for_user
    ms.delete_api_key(api_key=api_key.token)

    # test deletion
    ms.delete_user(user_1.id)
    ms.delete_user(user_2.id)
    ms.delete_agent(agent_1.id)
    ms.delete_source(source_1.id)
