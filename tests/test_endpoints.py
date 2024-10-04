import json
import os
import uuid

from letta import create_client
from letta.agent import Agent
from letta.embeddings import embedding_model
from letta.llm_api.llm_api_tools import create
from letta.prompts import gpt_system
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.message import Message
from tests.helpers.endpoints_helper import (
    agent_uuid,
    assert_contains_correct_inner_monologue,
    assert_contains_valid_function_call,
    assert_inner_monologue_is_present_and_valid,
    assert_invoked_function_call,
    assert_invoked_send_message_with_keyword,
    assert_sanity_checks,
    setup_agent,
)
from tests.helpers.utils import cleanup

messages = [Message(role="system", text=gpt_system.get_system_text("memgpt_chat")), Message(role="user", text="How are you?")]

# defaults (letta hosted)
embedding_config_path = "configs/embedding_model_configs/letta-hosted.json"
llm_config_path = "configs/llm_model_configs/letta-hosted.json"

# directories
embedding_config_dir = "configs/embedding_model_configs"
llm_config_dir = "configs/llm_model_configs"


def check_first_response_is_valid_for_llm_endpoint(filename: str, inner_thoughts_in_kwargs: bool = False):
    """
    Checks that the first response is valid:

    1. Contains either send_message or archival_memory_search
    2. Contains valid usage of the function
    3. Contains inner monologue

    Note: This is acting on the raw LLM response, note the usage of `create`
    """
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename, embedding_config_path)

    tools = [client.get_tool(client.get_tool_id(name=name)) for name in agent_state.tools]
    agent = Agent(
        interface=None,
        tools=tools,
        agent_state=agent_state,
    )

    response = create(
        llm_config=agent_state.llm_config,
        user_id=str(uuid.UUID(int=1)),  # dummy user_id
        messages=agent._messages,
        functions=agent.functions,
        functions_python=agent.functions_python,
    )

    # Basic check
    assert response is not None

    # Select first choice
    choice = response.choices[0]

    # Ensure that the first message returns a "send_message"
    validator_func = lambda function_call: function_call.name == "send_message" or function_call.name == "archival_memory_search"
    assert_contains_valid_function_call(choice.message, validator_func)

    # Assert that the message has an inner monologue
    assert_contains_correct_inner_monologue(choice, inner_thoughts_in_kwargs)


def check_response_contains_keyword(filename: str):
    """
    Checks that the prompted response from the LLM contains a chosen keyword

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename, embedding_config_path)

    keyword = "banana"
    keyword_message = f'This is a test to see if you can see my message. If you can see my message, please respond by calling send_message using a message that includes the word "{keyword}"'
    response = client.user_message(agent_id=agent_state.id, message=keyword_message)

    # Basic checks
    assert_sanity_checks(response)

    # Make sure the message was sent
    assert_invoked_send_message_with_keyword(response.messages, keyword)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)


def check_agent_uses_external_tool(filename: str):
    """
    Checks that the LLM will use external tools if instructed

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    from crewai_tools import ScrapeWebsiteTool

    from letta.schemas.tool import Tool

    crewai_tool = ScrapeWebsiteTool(website_url="https://www.example.com")
    tool = Tool.from_crewai(crewai_tool)
    tool_name = tool.name

    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    client.add_tool(tool)

    # Set up persona for tool usage
    persona = f"""

    My name is Letta.

    I am a personal assistant who answers a user's questions about a website `example.com`. When a user asks me a question about `example.com`, I will use a tool called {tool_name} which will search `example.com` and answer the relevant question.

    Don’t forget - inner monologue / inner thoughts should always be different than the contents of send_message! send_message is how you communicate with the user, whereas inner thoughts are your own personal inner thoughts.
    """

    agent_state = setup_agent(client, filename, embedding_config_path, memory_persona_str=persona, tools=[tool_name])

    response = client.user_message(agent_id=agent_state.id, message="What's on the example.com website?")

    # Basic checks
    assert_sanity_checks(response)

    # Make sure the tool was called
    assert_invoked_function_call(response.messages, tool_name)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)


def check_agent_recall_chat_memory(filename: str):
    """
    Checks that the LLM will recall the chat memory, specifically the human persona.

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    human_name = "BananaBoy"
    agent_state = setup_agent(client, filename, embedding_config_path, memory_human_str=f"My name is {human_name}")

    response = client.user_message(agent_id=agent_state.id, message="Repeat my name back to me.")

    # Basic checks
    assert_sanity_checks(response)

    # Make sure my name was repeated back to me
    assert_invoked_send_message_with_keyword(response.messages, human_name)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)


def check_agent_archival_memory_retrieval(filename: str):
    """
    Checks that the LLM will execute an archival memory retrieval.

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename, embedding_config_path)
    secret_word = "banana"
    client.insert_archival_memory(agent_state.id, f"The secret word is {secret_word}!")

    response = client.user_message(agent_id=agent_state.id, message="Search archival memory for the secret word and repeat it back to me.")

    # Basic checks
    assert_sanity_checks(response)

    # Make sure archival_memory_search was called
    assert_invoked_function_call(response.messages, "archival_memory_search")

    # Make sure secret was repeated back to me
    assert_invoked_send_message_with_keyword(response.messages, secret_word)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)


def run_embedding_endpoint(filename):
    # load JSON file
    config_data = json.load(open(filename, "r"))
    print(config_data)
    embedding_config = EmbeddingConfig(**config_data)
    model = embedding_model(embedding_config)
    query_text = "hello"
    query_vec = model.get_text_embedding(query_text)
    print("vector dim", len(query_vec))
    assert query_vec is not None


# ======================================================================================================================
# OPENAI TESTS
# ======================================================================================================================
def test_openai_gpt_4_returns_valid_first_message():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_openai_gpt_4_returns_keyword():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    check_response_contains_keyword(filename)


def test_openai_gpt_4_uses_external_tool():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    check_agent_uses_external_tool(filename)


def test_openai_gpt_4_recall_chat_memory():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    check_agent_recall_chat_memory(filename)


def test_openai_gpt_4_archival_memory_retrieval():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    check_agent_archival_memory_retrieval(filename)


def test_embedding_endpoint_openai():
    filename = os.path.join(embedding_config_dir, "text-embedding-ada-002.json")
    run_embedding_endpoint(filename)


# ======================================================================================================================
# LETTA HOSTED
# ======================================================================================================================
def test_llm_endpoint_letta_hosted():
    filename = os.path.join(llm_config_dir, "letta-hosted.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_letta_hosted():
    filename = os.path.join(embedding_config_dir, "letta-hosted.json")
    run_embedding_endpoint(filename)


# ======================================================================================================================
# LOCAL MODELS
# ======================================================================================================================
def test_embedding_endpoint_local():
    filename = os.path.join(embedding_config_dir, "local.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_ollama():
    filename = os.path.join(llm_config_dir, "ollama.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_ollama():
    filename = os.path.join(embedding_config_dir, "ollama.json")
    run_embedding_endpoint(filename)


# ======================================================================================================================
# ANTHROPIC TESTS
# ======================================================================================================================
def test_llm_endpoint_anthropic():
    filename = os.path.join(llm_config_dir, "anthropic.json")
    check_first_response_is_valid_for_llm_endpoint(filename)
