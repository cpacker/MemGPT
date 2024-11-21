import json
import logging
import uuid
from typing import Callable, List, Optional, Union

from letta.llm_api.helpers import unpack_inner_thoughts_from_kwargs
from letta.schemas.tool_rule import BaseToolRule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from letta import LocalClient, RESTClient, create_client
from letta.agent import Agent
from letta.config import LettaConfig
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.embeddings import embedding_model
from letta.errors import (
    InvalidFunctionCallError,
    InvalidInnerMonologueError,
    MissingFunctionCallError,
    MissingInnerMonologueError,
)
from letta.llm_api.llm_api_tools import create
from letta.local_llm.constants import INNER_THOUGHTS_KWARG
from letta.schemas.agent import AgentState
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.letta_message import (
    FunctionCallMessage,
    InternalMonologue,
    LettaMessage,
)
from letta.schemas.letta_response import LettaResponse
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
)
from letta.utils import get_human_text, get_persona_text, json_dumps
from tests.helpers.utils import cleanup

# Generate uuid for agent name for this example
namespace = uuid.NAMESPACE_DNS
agent_uuid = str(uuid.uuid5(namespace, "test-endpoints-agent"))

# defaults (letta hosted)
EMBEDDING_CONFIG_PATH = "tests/configs/embedding_model_configs/letta-hosted.json"
LLM_CONFIG_PATH = "tests/configs/llm_model_configs/letta-hosted.json"


# ======================================================================================================================
# Section: Test Setup
# These functions help setup the test
# ======================================================================================================================


def setup_agent(
    client: Union[LocalClient, RESTClient],
    filename: str,
    memory_human_str: str = get_human_text(DEFAULT_HUMAN),
    memory_persona_str: str = get_persona_text(DEFAULT_PERSONA),
    tools: Optional[List[str]] = None,
    tool_rules: Optional[List[BaseToolRule]] = None,
    agent_uuid: str = agent_uuid,
) -> AgentState:
    config_data = json.load(open(filename, "r"))
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(EMBEDDING_CONFIG_PATH)))

    # setup config
    config = LettaConfig()
    config.default_llm_config = llm_config
    config.default_embedding_config = embedding_config
    config.save()

    memory = ChatMemory(human=memory_human_str, persona=memory_persona_str)
    agent_state = client.create_agent(
        name=agent_uuid, llm_config=llm_config, embedding_config=embedding_config, memory=memory, tools=tools, tool_rules=tool_rules
    )

    return agent_state


# ======================================================================================================================
# Section: Complex E2E Tests
# These functions describe individual testing scenarios.
# ======================================================================================================================


def check_first_response_is_valid_for_llm_endpoint(filename: str) -> ChatCompletionResponse:
    """
    Checks that the first response is valid:

    1. Contains either send_message or archival_memory_search
    2. Contains valid usage of the function
    3. Contains inner monologue

    Note: This is acting on the raw LLM response, note the usage of `create`
    """
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename)

    tools = [client.get_tool(client.get_tool_id(name=name)) for name in agent_state.tools]
    agent = Agent(interface=None, tools=tools, agent_state=agent_state, user=client.user)

    response = create(
        llm_config=agent_state.llm_config,
        user_id=str(uuid.UUID(int=1)),  # dummy user_id
        messages=agent._messages,
        functions=agent.functions,
        functions_python=agent.functions_python,
    )

    # Basic check
    assert response is not None, response
    assert response.choices is not None, response
    assert len(response.choices) > 0, response
    assert response.choices[0] is not None, response

    # Select first choice
    choice = response.choices[0]

    # Ensure that the first message returns a "send_message"
    validator_func = lambda function_call: function_call.name == "send_message" or function_call.name == "archival_memory_search"
    assert_contains_valid_function_call(choice.message, validator_func)

    # Assert that the message has an inner monologue
    assert_contains_correct_inner_monologue(choice, agent_state.llm_config.put_inner_thoughts_in_kwargs)

    return response


def check_response_contains_keyword(filename: str, keyword="banana") -> LettaResponse:
    """
    Checks that the prompted response from the LLM contains a chosen keyword

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename)

    keyword_message = f'This is a test to see if you can see my message. If you can see my message, please respond by calling send_message using a message that includes the word "{keyword}"'
    response = client.user_message(agent_id=agent_state.id, message=keyword_message)

    # Basic checks
    assert_sanity_checks(response)

    # Make sure the message was sent
    assert_invoked_send_message_with_keyword(response.messages, keyword)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


def check_agent_uses_external_tool(filename: str) -> LettaResponse:
    """
    Checks that the LLM will use external tools if instructed

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    from composio_langchain import Action

    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    tool = client.load_composio_tool(action=Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER)
    tool_name = tool.name

    # Set up persona for tool usage
    persona = f"""

    My name is Letta.

    I am a personal assistant who answers a user's questions about a website `example.com`. When a user asks me a question about `example.com`, I will use a tool called {tool_name} which will search `example.com` and answer the relevant question.

    Don’t forget - inner monologue / inner thoughts should always be different than the contents of send_message! send_message is how you communicate with the user, whereas inner thoughts are your own personal inner thoughts.
    """

    agent_state = setup_agent(client, filename, memory_persona_str=persona, tools=[tool_name])

    response = client.user_message(agent_id=agent_state.id, message="What's on the example.com website?")

    # Basic checks
    assert_sanity_checks(response)

    # Make sure the tool was called
    assert_invoked_function_call(response.messages, tool_name)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


def check_agent_recall_chat_memory(filename: str) -> LettaResponse:
    """
    Checks that the LLM will recall the chat memory, specifically the human persona.

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    human_name = "BananaBoy"
    agent_state = setup_agent(client, filename, memory_human_str=f"My name is {human_name}")

    response = client.user_message(agent_id=agent_state.id, message="Repeat my name back to me.")

    # Basic checks
    assert_sanity_checks(response)

    # Make sure my name was repeated back to me
    assert_invoked_send_message_with_keyword(response.messages, human_name)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


def check_agent_archival_memory_insert(filename: str) -> LettaResponse:
    """
    Checks that the LLM will execute an archival memory insert.

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename)
    secret_word = "banana"

    response = client.user_message(
        agent_id=agent_state.id,
        message=f"Please insert the secret word '{secret_word}' into archival memory.",
    )

    # Basic checks
    assert_sanity_checks(response)

    # Make sure archival_memory_search was called
    assert_invoked_function_call(response.messages, "archival_memory_insert")

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


def check_agent_archival_memory_retrieval(filename: str) -> LettaResponse:
    """
    Checks that the LLM will execute an archival memory retrieval.

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)
    agent_state = setup_agent(client, filename)
    secret_word = "banana"
    client.insert_archival_memory(agent_state.id, f"The secret word is {secret_word}!")

    response = client.user_message(
        agent_id=agent_state.id,
        message="Search archival memory for the secret word. If you find it successfully, you MUST respond by using the `send_message` function with a message that includes the secret word so I know you found it.",
    )

    # Basic checks
    assert_sanity_checks(response)

    # Make sure archival_memory_search was called
    assert_invoked_function_call(response.messages, "archival_memory_search")

    # Make sure secret was repeated back to me
    assert_invoked_send_message_with_keyword(response.messages, secret_word)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


def check_agent_edit_core_memory(filename: str) -> LettaResponse:
    """
    Checks that the LLM is able to edit its core memories

    Note: This is acting on the Letta response, note the usage of `user_message`
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    human_name_a = "AngryAardvark"
    human_name_b = "BananaBoy"
    agent_state = setup_agent(client, filename, memory_human_str=f"My name is {human_name_a}")
    client.user_message(agent_id=agent_state.id, message=f"Actually, my name changed. It is now {human_name_b}")
    response = client.user_message(agent_id=agent_state.id, message="Repeat my name back to me.")

    # Basic checks
    assert_sanity_checks(response)

    # Make sure my name was repeated back to me
    assert_invoked_send_message_with_keyword(response.messages, human_name_b)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


def check_agent_summarize_memory_simple(filename: str) -> LettaResponse:
    """
    Checks that the LLM is able to summarize its memory
    """
    # Set up client
    client = create_client()
    cleanup(client=client, agent_uuid=agent_uuid)

    agent_state = setup_agent(client, filename)

    # Send a couple messages
    friend_name = "Shub"
    client.user_message(agent_id=agent_state.id, message="Hey, how's it going? What do you think about this whole shindig")
    client.user_message(agent_id=agent_state.id, message=f"By the way, my friend's name is {friend_name}!")
    client.user_message(agent_id=agent_state.id, message="Does the number 42 ring a bell?")

    # Summarize
    agent = client.server._get_or_load_agent(agent_id=agent_state.id)
    agent.summarize_messages_inplace()
    print(f"Summarization succeeded: messages[1] = \n\n{json_dumps(agent.messages[1])}\n")

    response = client.user_message(agent_id=agent_state.id, message="What is my friend's name?")
    # Basic checks
    assert_sanity_checks(response)

    # Make sure my name was repeated back to me
    assert_invoked_send_message_with_keyword(response.messages, friend_name)

    # Make sure some inner monologue is present
    assert_inner_monologue_is_present_and_valid(response.messages)

    return response


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
# Section: Letta Message Assertions
# These functions are validating elements of parsed Letta Messsage
# ======================================================================================================================


def assert_sanity_checks(response: LettaResponse):
    assert response is not None, response
    assert response.messages is not None, response
    assert len(response.messages) > 0, response


def assert_invoked_send_message_with_keyword(messages: List[LettaMessage], keyword: str, case_sensitive: bool = False) -> None:
    # Find first instance of send_message
    target_message = None
    for message in messages:
        if isinstance(message, FunctionCallMessage) and message.function_call.name == "send_message":
            target_message = message
            break

    # No messages found with `send_messages`
    if target_message is None:
        raise MissingFunctionCallError(messages=messages, explanation="Missing `send_message` function call")

    send_message_function_call = target_message.function_call
    try:
        arguments = json.loads(send_message_function_call.arguments)
    except:
        raise InvalidFunctionCallError(messages=[target_message], explanation="Function call arguments could not be loaded into JSON")

    # Message field not in send_message
    if "message" not in arguments:
        raise InvalidFunctionCallError(
            messages=[target_message], explanation=f"send_message function call does not have required field `message`"
        )

    # Check that the keyword is in the message arguments
    if not case_sensitive:
        keyword = keyword.lower()
        arguments["message"] = arguments["message"].lower()

    if not keyword in arguments["message"]:
        raise InvalidFunctionCallError(messages=[target_message], explanation=f"Message argument did not contain keyword={keyword}")


def assert_invoked_function_call(messages: List[LettaMessage], function_name: str) -> None:
    for message in messages:
        if isinstance(message, FunctionCallMessage) and message.function_call.name == function_name:
            # Found it, do nothing
            return

    raise MissingFunctionCallError(
        messages=messages, explanation=f"No messages were found invoking function call with name: {function_name}"
    )


def assert_inner_monologue_is_present_and_valid(messages: List[LettaMessage]) -> None:
    for message in messages:
        if isinstance(message, InternalMonologue):
            # Found it, do nothing
            return

    raise MissingInnerMonologueError(messages=messages)


# ======================================================================================================================
# Section: Raw API Assertions
# These functions are validating elements of the (close to) raw LLM API's response
# ======================================================================================================================


def assert_contains_valid_function_call(
    message: Message,
    function_call_validator: Optional[Callable[[FunctionCall], bool]] = None,
    validation_failure_summary: Optional[str] = None,
) -> None:
    """
    Helper function to check that a message contains a valid function call.

    There is an Optional parameter `function_call_validator` that specifies a validator function.
    This function gets called on the resulting function_call to validate the function is what we expect.
    """
    if (hasattr(message, "function_call") and message.function_call is not None) and (
        hasattr(message, "tool_calls") and message.tool_calls is not None
    ):
        raise InvalidFunctionCallError(messages=[message], explanation="Both function_call and tool_calls is present in the message")
    elif hasattr(message, "function_call") and message.function_call is not None:
        function_call = message.function_call
    elif hasattr(message, "tool_calls") and message.tool_calls is not None:
        # Note: We only take the first one for now. Is this a problem? @charles
        # This seems to be standard across the repo
        function_call = message.tool_calls[0].function
    else:
        # Throw a missing function call error
        raise MissingFunctionCallError(messages=[message])

    if function_call_validator and not function_call_validator(function_call):
        raise InvalidFunctionCallError(messages=[message], explanation=validation_failure_summary)


def assert_inner_monologue_is_valid(message: Message) -> None:
    """
    Helper function to check that the inner monologue is valid.
    """
    # Sometimes the syntax won't be correct and internal syntax will leak into message
    invalid_phrases = ["functions", "send_message", "arguments"]

    monologue = message.content
    for phrase in invalid_phrases:
        if phrase in monologue:
            raise InvalidInnerMonologueError(messages=[message], explanation=f"{phrase} is in monologue")


def assert_contains_correct_inner_monologue(choice: Choice, inner_thoughts_in_kwargs: bool) -> None:
    """
    Helper function to check that the inner monologue exists and is valid.
    """
    # Unpack inner thoughts out of function kwargs, and repackage into choice
    if inner_thoughts_in_kwargs:
        choice = unpack_inner_thoughts_from_kwargs(choice, INNER_THOUGHTS_KWARG)

    message = choice.message
    monologue = message.content
    if not monologue or monologue is None or monologue == "":
        raise MissingInnerMonologueError(messages=[message])

    assert_inner_monologue_is_valid(message)
