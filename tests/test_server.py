import json
import uuid
import warnings
from typing import List, Tuple

import pytest

import letta.utils as utils
from letta.constants import BASE_MEMORY_TOOLS, BASE_TOOLS
from letta.schemas.block import CreateBlock
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message import (
    FunctionCallMessage,
    FunctionReturn,
    InternalMonologue,
    LettaMessage,
    SystemMessage,
    UserMessage,
)
from letta.schemas.user import User

from .test_managers import DEFAULT_EMBEDDING_CONFIG

utils.DEBUG = True
from letta.config import LettaConfig
from letta.schemas.agent import CreateAgent
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.source import Source
from letta.server.server import SyncServer

from .utils import DummyDataConnector


@pytest.fixture(scope="module")
def server():
    config = LettaConfig.load()
    print("CONFIG PATH", config.config_path)

    config.save()

    server = SyncServer()
    return server


@pytest.fixture(scope="module")
def org_id(server):
    # create org
    org = server.organization_manager.create_default_organization()
    print(f"Created org\n{org.id}")

    yield org.id

    # cleanup
    server.organization_manager.delete_organization_by_id(org.id)


@pytest.fixture(scope="module")
def user_id(server, org_id):
    # create user
    user = server.user_manager.create_default_user()
    print(f"Created user\n{user.id}")

    yield user.id

    # cleanup
    server.user_manager.delete_user_by_id(user.id)


@pytest.fixture(scope="module")
def base_tools(server, user_id):
    actor = server.user_manager.get_user_or_default(user_id)
    tools = []
    for tool_name in BASE_TOOLS:
        tools.append(server.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor))

    yield tools


@pytest.fixture(scope="module")
def base_memory_tools(server, user_id):
    actor = server.user_manager.get_user_or_default(user_id)
    tools = []
    for tool_name in BASE_MEMORY_TOOLS:
        tools.append(server.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor))

    yield tools


@pytest.fixture(scope="module")
def agent_id(server, user_id, base_tools):
    # create agent
    actor = server.user_manager.get_user_or_default(user_id)
    agent_state = server.create_agent(
        request=CreateAgent(
            name="test_agent",
            tool_ids=[t.id for t in base_tools],
            memory_blocks=[],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=actor,
    )
    print(f"Created agent\n{agent_state}")
    yield agent_state.id

    # cleanup
    server.agent_manager.delete_agent(agent_state.id, actor=actor)


def test_error_on_nonexistent_agent(server, user_id, agent_id):
    try:
        fake_agent_id = str(uuid.uuid4())
        server.user_message(user_id=user_id, agent_id=fake_agent_id, message="Hello?")
        raise Exception("user_message call should have failed")
    except (KeyError, ValueError) as e:
        # Error is expected
        print(e)
    except:
        raise


@pytest.mark.order(1)
def test_user_message_memory(server, user_id, agent_id):
    try:
        server.user_message(user_id=user_id, agent_id=agent_id, message="/memory")
        raise Exception("user_message call should have failed")
    except ValueError as e:
        # Error is expected
        print(e)
    except:
        raise

    server.run_command(user_id=user_id, agent_id=agent_id, command="/memory")


@pytest.mark.order(3)
def test_load_data(server, user_id, agent_id):
    # create source
    passages_before = server.get_agent_archival(user_id=user_id, agent_id=agent_id, cursor=None, limit=10000)
    assert len(passages_before) == 0

    source = server.source_manager.create_source(
        Source(name="test_source", embedding_config=DEFAULT_EMBEDDING_CONFIG), actor=server.default_user
    )

    # load data
    archival_memories = [
        "alpha",
        "Cinderella wore a blue dress",
        "Dog eat dog",
        "ZZZ",
        "Shishir loves indian food",
    ]
    connector = DummyDataConnector(archival_memories)
    server.load_data(user_id, connector, source.name, agent_id=agent_id)

    # @pytest.mark.order(3)
    # def test_attach_source_to_agent(server, user_id, agent_id):
    # check archival memory size

    # attach source
    server.attach_source_to_agent(user_id=user_id, agent_id=agent_id, source_name="test_source")

    # check archival memory size
    passages_after = server.get_agent_archival(user_id=user_id, agent_id=agent_id, cursor=None, limit=10000)
    assert len(passages_after) == 5


def test_save_archival_memory(server, user_id, agent_id):
    # TODO: insert into archival memory
    pass


@pytest.mark.order(4)
def test_user_message(server, user_id, agent_id):
    # add data into recall memory
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    # server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    # server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    # server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")
    # server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")


@pytest.mark.order(5)
def test_get_recall_memory(server, org_id, user_id, agent_id):
    # test recall memory cursor pagination
    messages_1 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, limit=2)
    cursor1 = messages_1[-1].id
    messages_2 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, after=cursor1, limit=1000)
    messages_2[-1].id
    messages_3 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, limit=1000)
    messages_3[-1].id
    assert messages_3[-1].created_at >= messages_3[0].created_at
    assert len(messages_3) == len(messages_1) + len(messages_2)
    messages_4 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, reverse=True, before=cursor1)
    assert len(messages_4) == 1

    # test in-context message ids
    in_context_ids = server.get_in_context_message_ids(agent_id=agent_id)
    message_ids = [m.id for m in messages_3]
    for message_id in in_context_ids:
        assert message_id in message_ids, f"{message_id} not in {message_ids}"


# TODO: Out-of-date test. pagination commands are off
# @pytest.mark.order(6)
# def test_get_archival_memory(server, user_id, agent_id):
#     # test archival memory cursor pagination
#     passages_1 = server.get_agent_archival_cursor(user_id=user_id, agent_id=agent_id, reverse=False, limit=2, order_by="text")
#     assert len(passages_1) == 2, f"Returned {[p.text for p in passages_1]}, not equal to 2"
#     cursor1 = passages_1[-1].id
#     passages_2 = server.get_agent_archival_cursor(
#         user_id=user_id,
#         agent_id=agent_id,
#         reverse=False,
#         after=cursor1,
#         order_by="text",
#     )
#     cursor2 = passages_2[-1].id
#     passages_3 = server.get_agent_archival_cursor(
#         user_id=user_id,
#         agent_id=agent_id,
#         reverse=False,
#         before=cursor2,
#         limit=1000,
#         order_by="text",
#     )
#     passages_3[-1].id
#     # assert passages_1[0].text == "Cinderella wore a blue dress"
#     assert len(passages_2) in [3, 4]  # NOTE: exact size seems non-deterministic, so loosen test
#     assert len(passages_3) in [4, 5]  # NOTE: exact size seems non-deterministic, so loosen test

#     # test archival memory
#     passage_1 = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=1)
#     assert len(passage_1) == 1
#     passage_2 = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=1, count=1000)
#     assert len(passage_2) in [4, 5]  # NOTE: exact size seems non-deterministic, so loosen test
#     # test safe empty return
#     passage_none = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=1000, count=1000)
#     assert len(passage_none) == 0


def test_agent_rethink_rewrite_retry(server, user_id, agent_id):
    """Test the /rethink, /rewrite, and /retry commands in the CLI

    - "rethink" replaces the inner thoughts of the last assistant message
    - "rewrite" replaces the text of the last assistant message
    - "retry" retries the last assistant message
    """

    # Send an initial message
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")

    # Grab the raw Agent object
    letta_agent = server.load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]

    # Try "rethink"
    new_thought = "I am thinking about the meaning of life, the universe, and everything. Bananas?"
    assert last_agent_message.text is not None and last_agent_message.text != new_thought
    server.rethink_agent_message(agent_id=agent_id, new_thought=new_thought)

    # Grab the agent object again (make sure it's live)
    letta_agent = server.load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]
    assert last_agent_message.text == new_thought

    # Try "rewrite"
    assert last_agent_message.tool_calls is not None
    assert last_agent_message.tool_calls[0].function.name == "send_message"
    assert last_agent_message.tool_calls[0].function.arguments is not None
    args_json = json.loads(last_agent_message.tool_calls[0].function.arguments)
    assert "message" in args_json and args_json["message"] is not None and args_json["message"] != ""

    new_text = "Why hello there my good friend! Is 42 what you're looking for? Bananas?"
    server.rewrite_agent_message(agent_id=agent_id, new_text=new_text)

    # Grab the agent object again (make sure it's live)
    letta_agent = server.load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]
    args_json = json.loads(last_agent_message.tool_calls[0].function.arguments)
    assert "message" in args_json and args_json["message"] is not None and args_json["message"] == new_text

    # Try retry
    server.retry_agent_message(agent_id=agent_id)

    # Grab the agent object again (make sure it's live)
    letta_agent = server.load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]

    # Make sure the inner thoughts changed
    assert last_agent_message.text is not None and last_agent_message.text != new_thought

    # Make sure the message changed
    args_json = json.loads(last_agent_message.tool_calls[0].function.arguments)
    print(args_json)
    assert "message" in args_json and args_json["message"] is not None and args_json["message"] != new_text


def test_get_context_window_overview(server: SyncServer, user_id: str, agent_id: str):
    """Test that the context window overview fetch works"""

    overview = server.get_agent_context_window(user_id=user_id, agent_id=agent_id)
    assert overview is not None

    # Run some basic checks
    assert overview.context_window_size_max is not None
    assert overview.context_window_size_current is not None
    assert overview.num_archival_memory is not None
    assert overview.num_recall_memory is not None
    assert overview.num_tokens_external_memory_summary is not None
    assert overview.num_tokens_system is not None
    assert overview.system_prompt is not None
    assert overview.num_tokens_core_memory is not None
    assert overview.core_memory is not None
    assert overview.num_tokens_summary_memory is not None
    if overview.num_tokens_summary_memory > 0:
        assert overview.summary_memory is not None
    else:
        assert overview.summary_memory is None
    assert overview.num_tokens_functions_definitions is not None
    if overview.num_tokens_functions_definitions > 0:
        assert overview.functions_definitions is not None
    else:
        assert overview.functions_definitions is None
    assert overview.num_tokens_messages is not None
    assert overview.messages is not None

    assert overview.context_window_size_max >= overview.context_window_size_current
    assert overview.context_window_size_current == (
        overview.num_tokens_system
        + overview.num_tokens_core_memory
        + overview.num_tokens_summary_memory
        + overview.num_tokens_messages
        + overview.num_tokens_functions_definitions
        + overview.num_tokens_external_memory_summary
    )


def test_delete_agent_same_org(server: SyncServer, org_id: str, user_id: str):
    agent_state = server.create_agent(
        request=CreateAgent(
            name="nonexistent_tools_agent",
            memory_blocks=[],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=server.user_manager.get_user_or_default(user_id),
    )

    # create another user in the same org
    another_user = server.user_manager.create_user(User(organization_id=org_id, name="another"))

    # test that another user in the same org can delete the agent
    server.agent_manager.delete_agent(agent_state.id, actor=another_user)


def _test_get_messages_letta_format(
    server,
    user_id,
    agent_id,
    reverse=False,
):
    """Test mapping between messages and letta_messages with reverse=False."""

    messages = server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        limit=1000,
        reverse=reverse,
        return_message_object=True,
    )
    assert all(isinstance(m, Message) for m in messages)

    letta_messages = server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        limit=1000,
        reverse=reverse,
        return_message_object=False,
    )
    assert all(isinstance(m, LettaMessage) for m in letta_messages)

    print(f"Messages: {len(messages)}, LettaMessages: {len(letta_messages)}")

    letta_message_index = 0
    for i, message in enumerate(messages):
        assert isinstance(message, Message)

        # Defensive bounds check for letta_messages
        if letta_message_index >= len(letta_messages):
            print(f"Error: letta_message_index out of range. Expected more letta_messages for message {i}: {message.role}")
            raise ValueError(f"Mismatch in letta_messages length. Index: {letta_message_index}, Length: {len(letta_messages)}")

        print(f"Processing message {i}: {message.role}, {message.text[:50] if message.text else 'null'}")
        while letta_message_index < len(letta_messages):
            letta_message = letta_messages[letta_message_index]

            # Validate mappings for assistant role
            if message.role == MessageRole.assistant:
                print(f"Assistant Message at {i}: {type(letta_message)}")

                if reverse:
                    # Reverse handling: FunctionCallMessages come first
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            try:
                                json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                warnings.warn(f"Invalid JSON in function arguments: {tool_call.function.arguments}")
                            assert isinstance(letta_message, FunctionCallMessage)
                            letta_message_index += 1
                            if letta_message_index >= len(letta_messages):
                                break
                            letta_message = letta_messages[letta_message_index]

                    if message.text:
                        assert isinstance(letta_message, InternalMonologue)
                        letta_message_index += 1
                    else:
                        assert message.tool_calls is not None

                else:  # Non-reverse handling
                    if message.text:
                        assert isinstance(letta_message, InternalMonologue)
                        letta_message_index += 1
                        if letta_message_index >= len(letta_messages):
                            break
                        letta_message = letta_messages[letta_message_index]

                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            try:
                                json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                warnings.warn(f"Invalid JSON in function arguments: {tool_call.function.arguments}")
                            assert isinstance(letta_message, FunctionCallMessage)
                            assert tool_call.function.name == letta_message.function_call.name
                            assert tool_call.function.arguments == letta_message.function_call.arguments
                            letta_message_index += 1
                            if letta_message_index >= len(letta_messages):
                                break
                            letta_message = letta_messages[letta_message_index]

            elif message.role == MessageRole.user:
                assert isinstance(letta_message, UserMessage)
                assert message.text == letta_message.message
                letta_message_index += 1

            elif message.role == MessageRole.system:
                assert isinstance(letta_message, SystemMessage)
                assert message.text == letta_message.message
                letta_message_index += 1

            elif message.role == MessageRole.tool:
                assert isinstance(letta_message, FunctionReturn)
                assert message.text == letta_message.function_return
                letta_message_index += 1

            else:
                raise ValueError(f"Unexpected message role: {message.role}")

            break  # Exit the letta_messages loop after processing one mapping

    if letta_message_index < len(letta_messages):
        warnings.warn(f"Extra letta_messages found: {len(letta_messages) - letta_message_index}")


def test_get_messages_letta_format(server, user_id, agent_id):
    # for reverse in [False, True]:
    for reverse in [False]:
        _test_get_messages_letta_format(server, user_id, agent_id, reverse=reverse)


EXAMPLE_TOOL_SOURCE = '''
def ingest(message: str):
    """
    Ingest a message into the system.

    Args:
        message (str): The message to ingest into the system.

    Returns:
        str: The result of ingesting the message.
    """
    return f"Ingested message {message}"

'''


EXAMPLE_TOOL_SOURCE_WITH_DISTRACTOR = '''
def util_do_nothing():
    """
    A util function that does nothing.

    Returns:
        str: Dummy output.
    """
    print("I'm a distractor")

def ingest(message: str):
    """
    Ingest a message into the system.

    Args:
        message (str): The message to ingest into the system.

    Returns:
        str: The result of ingesting the message.
    """
    util_do_nothing()
    return f"Ingested message {message}"

'''


def test_tool_run(server, mock_e2b_api_key_none, user_id, agent_id):
    """Test that the server can run tools"""

    result = server.run_tool_from_source(
        user_id=user_id,
        tool_source=EXAMPLE_TOOL_SOURCE,
        tool_source_type="python",
        tool_args=json.dumps({"message": "Hello, world!"}),
        # tool_name="ingest",
    )
    print(result)
    assert result.status == "success"
    assert result.function_return == "Ingested message Hello, world!", result.function_return
    assert result.stdout == [""]
    assert result.stderr == [""]

    result = server.run_tool_from_source(
        user_id=user_id,
        tool_source=EXAMPLE_TOOL_SOURCE,
        tool_source_type="python",
        tool_args=json.dumps({"message": "Well well well"}),
        # tool_name="ingest",
    )
    print(result)
    assert result.status == "success"
    assert result.function_return == "Ingested message Well well well", result.function_return
    assert result.stdout == [""]
    assert result.stderr == [""]

    result = server.run_tool_from_source(
        user_id=user_id,
        tool_source=EXAMPLE_TOOL_SOURCE,
        tool_source_type="python",
        tool_args=json.dumps({"bad_arg": "oh no"}),
        # tool_name="ingest",
    )
    print(result)
    assert result.status == "error"
    assert "Error" in result.function_return, result.function_return
    assert "missing 1 required positional argument" in result.function_return, result.function_return
    assert result.stdout == [""]
    assert result.stderr != [""], "missing 1 required positional argument" in result.stderr[0]

    # Test that we can still pull the tool out by default (pulls that last tool in the source)
    result = server.run_tool_from_source(
        user_id=user_id,
        tool_source=EXAMPLE_TOOL_SOURCE_WITH_DISTRACTOR,
        tool_source_type="python",
        tool_args=json.dumps({"message": "Well well well"}),
        # tool_name="ingest",
    )
    print(result)
    assert result.status == "success"
    assert result.function_return == "Ingested message Well well well", result.function_return
    assert result.stdout != [""], "I'm a distractor" in result.stdout[0]
    assert result.stderr == [""]

    # Test that we can pull the tool out by name
    result = server.run_tool_from_source(
        user_id=user_id,
        tool_source=EXAMPLE_TOOL_SOURCE_WITH_DISTRACTOR,
        tool_source_type="python",
        tool_args=json.dumps({"message": "Well well well"}),
        tool_name="ingest",
    )
    print(result)
    assert result.status == "success"
    assert result.function_return == "Ingested message Well well well", result.function_return
    assert result.stdout != [""], "I'm a distractor" in result.stdout[0]
    assert result.stderr == [""]

    # Test that we can pull a different tool out by name
    result = server.run_tool_from_source(
        user_id=user_id,
        tool_source=EXAMPLE_TOOL_SOURCE_WITH_DISTRACTOR,
        tool_source_type="python",
        tool_args=json.dumps({}),
        tool_name="util_do_nothing",
    )
    print(result)
    assert result.status == "success"
    assert result.function_return == str(None), result.function_return
    assert result.stdout != [""], "I'm a distractor" in result.stdout[0]
    assert result.stderr == [""]


def test_composio_client_simple(server):
    apps = server.get_composio_apps()
    # Assert there's some amount of apps returned
    assert len(apps) > 0

    app = apps[0]
    actions = server.get_composio_actions_from_app_name(composio_app_name=app.name)

    # Assert there's some amount of actions
    assert len(actions) > 0


def test_memory_rebuild_count(server, user_id, mock_e2b_api_key_none, base_tools, base_memory_tools):
    """Test that the memory rebuild is generating the correct number of role=system messages"""
    actor = server.user_manager.get_user_or_default(user_id)
    # create agent
    agent_state = server.create_agent(
        request=CreateAgent(
            name="memory_rebuild_test_agent",
            tool_ids=[t.id for t in base_tools + base_memory_tools],
            memory_blocks=[
                CreateBlock(label="human", value="The human's name is Bob."),
                CreateBlock(label="persona", value="My name is Alice."),
            ],
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
        ),
        actor=actor,
    )
    print(f"Created agent\n{agent_state}")

    def count_system_messages_in_recall() -> Tuple[int, List[LettaMessage]]:

        # At this stage, there should only be 1 system message inside of recall storage
        letta_messages = server.get_agent_recall_cursor(
            user_id=user_id,
            agent_id=agent_state.id,
            limit=1000,
            # reverse=reverse,
            return_message_object=False,
        )
        assert all(isinstance(m, LettaMessage) for m in letta_messages)

        print("LETTA_MESSAGES:")
        for i, m in enumerate(letta_messages):
            print(f"{i}: {type(m)} ...{str(m)[-50:]}")

        # Collect system messages and their texts
        system_messages = [m for m in letta_messages if m.message_type == "system_message"]
        return len(system_messages), letta_messages

    try:
        # At this stage, there should only be 1 system message inside of recall storage
        num_system_messages, all_messages = count_system_messages_in_recall()
        assert num_system_messages == 1, (num_system_messages, all_messages)

        # Assuming core memory append actually ran correctly, at this point there should be 2 messages
        server.user_message(user_id=user_id, agent_id=agent_state.id, message="Append 'banana' to your core memory")

        # At this stage, there should be 2 system message inside of recall storage
        num_system_messages, all_messages = count_system_messages_in_recall()
        assert num_system_messages == 2, (num_system_messages, all_messages)

        # Run server.load_agent, and make sure that the number of system messages is still 2
        server.load_agent(agent_id=agent_state.id)

        num_system_messages, all_messages = count_system_messages_in_recall()
        assert num_system_messages == 2, (num_system_messages, all_messages)

    finally:
        # cleanup
        server.agent_manager.delete_agent(agent_state.id, actor=actor)
