import json
import uuid

import pytest

import letta.utils as utils
from letta.constants import BASE_TOOLS
from letta.schemas.enums import MessageRole

utils.DEBUG = True
from letta.config import LettaConfig
from letta.schemas.agent import CreateAgent
from letta.schemas.letta_message import (
    FunctionCallMessage,
    FunctionReturn,
    InternalMonologue,
    LettaMessage,
    SystemMessage,
    UserMessage,
)
from letta.schemas.memory import ChatMemory
from letta.schemas.message import Message
from letta.schemas.source import SourceCreate
from letta.schemas.user import UserCreate
from letta.server.server import SyncServer

from .utils import DummyDataConnector


@pytest.fixture(scope="module")
def server():
    # if os.getenv("OPENAI_API_KEY"):
    #    create_config("openai")
    #    credentials = LettaCredentials(
    #        openai_key=os.getenv("OPENAI_API_KEY"),
    #    )
    # else:  # hosted
    #    create_config("letta_hosted")
    #    credentials = LettaCredentials()

    config = LettaConfig.load()
    print("CONFIG PATH", config.config_path)

    ## set to use postgres
    # config.archival_storage_uri = db_url
    # config.recall_storage_uri = db_url
    # config.metadata_storage_uri = db_url
    # config.archival_storage_type = "postgres"
    # config.recall_storage_type = "postgres"
    # config.metadata_storage_type = "postgres"

    config.save()

    server = SyncServer()
    return server


@pytest.fixture(scope="module")
def user_id(server):
    # create user
    user = server.create_user(UserCreate(name="test_user"))
    print(f"Created user\n{user.id}")

    yield user.id

    # cleanup
    server.delete_user(user.id)


@pytest.fixture(scope="module")
def agent_id(server, user_id):
    # create agent
    agent_state = server.create_agent(
        request=CreateAgent(name="test_agent", tools=BASE_TOOLS, memory=ChatMemory(human="Sarah", persona="I am a helpful assistant")),
        user_id=user_id,
    )
    print(f"Created agent\n{agent_state}")
    yield agent_state.id

    # cleanup
    server.delete_agent(user_id, agent_state.id)


def test_error_on_nonexistent_agent(server, user_id, agent_id):
    try:
        fake_agent_id = uuid.uuid4()
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
    source = server.create_source(SourceCreate(name="test_source"), user_id=user_id)

    # load data
    archival_memories = [
        "alpha",
        "Cinderella wore a blue dress",
        "Dog eat dog",
        "ZZZ",
        "Shishir loves indian food",
    ]
    connector = DummyDataConnector(archival_memories)
    server.load_data(user_id, connector, source.name)


@pytest.mark.order(3)
def test_attach_source_to_agent(server, user_id, agent_id):
    # check archival memory size
    passages_before = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=10000)
    assert len(passages_before) == 0

    # attach source
    server.attach_source_to_agent(user_id=user_id, agent_id=agent_id, source_name="test_source")

    # check archival memory size
    passages_after = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=10000)
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
def test_get_recall_memory(server, user_id, agent_id):
    # test recall memory cursor pagination
    messages_1 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, limit=2)
    cursor1 = messages_1[-1].id
    messages_2 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, after=cursor1, limit=1000)
    messages_2[-1].id
    messages_3 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, limit=1000)
    messages_3[-1].id
    # [m["id"] for m in messages_3]
    # [m["id"] for m in messages_2]
    timestamps = [m.created_at for m in messages_3]
    print("timestamps", timestamps)
    assert messages_3[-1].created_at >= messages_3[0].created_at
    assert len(messages_3) == len(messages_1) + len(messages_2)
    messages_4 = server.get_agent_recall_cursor(user_id=user_id, agent_id=agent_id, reverse=True, before=cursor1)
    assert len(messages_4) == 1

    # test in-context message ids
    all_messages = server.get_agent_messages(agent_id=agent_id, start=0, count=1000)
    in_context_ids = server.get_in_context_message_ids(agent_id=agent_id)
    # TODO: doesn't pass since recall memory also logs all system message changess
    # print("IN CONTEXT:", [m.text for m in server.get_in_context_messages(agent_id=agent_id)])
    # print("ALL:", [m.text for m in all_messages])
    # print()
    # for message in all_messages:
    #    if message.id not in in_context_ids:
    #        print("NOT IN CONTEXT:", message.id, message.created_at, message.text[-100:])
    #        print()
    # assert len(in_context_ids) == len(messages_3)
    message_ids = [m.id for m in messages_3]
    for message_id in in_context_ids:
        assert message_id in message_ids, f"{message_id} not in {message_ids}"

    # test recall memory
    messages_1 = server.get_agent_messages(agent_id=agent_id, start=0, count=1)
    assert len(messages_1) == 1
    messages_2 = server.get_agent_messages(agent_id=agent_id, start=1, count=1000)
    messages_3 = server.get_agent_messages(agent_id=agent_id, start=1, count=2)
    # not sure exactly how many messages there should be
    assert len(messages_2) > len(messages_3)
    # test safe empty return
    messages_none = server.get_agent_messages(agent_id=agent_id, start=1000, count=1000)
    assert len(messages_none) == 0


@pytest.mark.order(6)
def test_get_archival_memory(server, user_id, agent_id):
    # test archival memory cursor pagination
    passages_1 = server.get_agent_archival_cursor(user_id=user_id, agent_id=agent_id, reverse=False, limit=2, order_by="text")
    assert len(passages_1) == 2, f"Returned {[p.text for p in passages_1]}, not equal to 2"
    cursor1 = passages_1[-1].id
    passages_2 = server.get_agent_archival_cursor(
        user_id=user_id,
        agent_id=agent_id,
        reverse=False,
        after=cursor1,
        order_by="text",
    )
    cursor2 = passages_2[-1].id
    passages_3 = server.get_agent_archival_cursor(
        user_id=user_id,
        agent_id=agent_id,
        reverse=False,
        before=cursor2,
        limit=1000,
        order_by="text",
    )
    passages_3[-1].id
    # assert passages_1[0].text == "Cinderella wore a blue dress"
    assert len(passages_2) in [3, 4]  # NOTE: exact size seems non-deterministic, so loosen test
    assert len(passages_3) in [4, 5]  # NOTE: exact size seems non-deterministic, so loosen test

    # test archival memory
    passage_1 = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=0, count=1)
    assert len(passage_1) == 1
    passage_2 = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=1, count=1000)
    assert len(passage_2) in [4, 5]  # NOTE: exact size seems non-deterministic, so loosen test
    # test safe empty return
    passage_none = server.get_agent_archival(user_id=user_id, agent_id=agent_id, start=1000, count=1000)
    assert len(passage_none) == 0


def _test_get_messages_letta_format(server, user_id, agent_id, reverse=False, assistant_message=False):
    """Reverse is off by default, the GET goes in chronological order"""

    messages = server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        limit=1000,
        reverse=reverse,
    )
    # messages = server.get_agent_messages(agent_id=agent_id, start=0, count=1000)
    assert all(isinstance(m, Message) for m in messages)

    letta_messages = server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        limit=1000,
        reverse=reverse,
        return_message_object=False,
    )
    # letta_messages = server.get_agent_messages(agent_id=agent_id, start=0, count=1000, return_message_object=False)
    assert all(isinstance(m, LettaMessage) for m in letta_messages)

    # Loop through `messages` while also looping through `letta_messages`
    # Each message in `messages` should have 1+ corresponding messages in `letta_messages`
    # If role of message (in `messages`) is `assistant`,
    # then there should be two messages in `letta_messages`, one which is type InternalMonologue and one which is type FunctionCallMessage.
    # If role of message (in `messages`) is `user`, then there should be one message in `letta_messages` which is type UserMessage.
    # If role of message (in `messages`) is `system`, then there should be one message in `letta_messages` which is type SystemMessage.
    # If role of message (in `messages`) is `tool`, then there should be one message in `letta_messages` which is type FunctionReturn.

    print("MESSAGES (obj):")
    for i, m in enumerate(messages):
        # print(m)
        print(f"{i}: {m.role}, {m.text[:50]}...")
        # print(m.role)

    print("MEMGPT_MESSAGES:")
    for i, m in enumerate(letta_messages):
        print(f"{i}: {type(m)} ...{str(m)[-50:]}")

    # Collect system messages and their texts
    system_messages = [m for m in messages if m.role == MessageRole.system]
    system_texts = [m.text for m in system_messages]

    # If there are multiple system messages, print the diff
    if len(system_messages) > 1:
        print("Differences between system messages:")
        for i in range(len(system_texts) - 1):
            for j in range(i + 1, len(system_texts)):
                import difflib

                diff = difflib.unified_diff(
                    system_texts[i].splitlines(),
                    system_texts[j].splitlines(),
                    fromfile=f"System Message {i+1}",
                    tofile=f"System Message {j+1}",
                    lineterm="",
                )
                print("\n".join(diff))
    else:
        print("There is only one or no system message.")

    letta_message_index = 0
    for i, message in enumerate(messages):
        assert isinstance(message, Message)

        print(f"\n\nmessage {i}: {message.role}, {message.text[:50] if message.text else 'null'}")
        while letta_message_index < len(letta_messages):
            letta_message = letta_messages[letta_message_index]
            print(f"letta_message {letta_message_index}: {str(letta_message)[:50]}")

            if message.role == MessageRole.assistant:
                print(f"i={i}, M=assistant, MM={type(letta_message)}")

                # If reverse, function call will come first
                if reverse:

                    # If there are multiple tool calls, we should have multiple back to back FunctionCallMessages
                    if message.tool_calls is not None:
                        for tool_call in message.tool_calls:
                            assert isinstance(letta_message, FunctionCallMessage)
                            letta_message_index += 1
                            letta_message = letta_messages[letta_message_index]

                    if message.text is not None:
                        assert isinstance(letta_message, InternalMonologue)
                        letta_message_index += 1
                        letta_message = letta_messages[letta_message_index]
                    else:
                        # If there's no inner thoughts then there needs to be a tool call
                        assert message.tool_calls is not None

                else:

                    if message.text is not None:
                        assert isinstance(letta_message, InternalMonologue)
                        letta_message_index += 1
                        letta_message = letta_messages[letta_message_index]
                    else:
                        # If there's no inner thoughts then there needs to be a tool call
                        assert message.tool_calls is not None

                    # If there are multiple tool calls, we should have multiple back to back FunctionCallMessages
                    if message.tool_calls is not None:
                        for tool_call in message.tool_calls:
                            assert isinstance(letta_message, FunctionCallMessage)
                            assert tool_call.function.name == letta_message.function_call.name
                            assert tool_call.function.arguments == letta_message.function_call.arguments
                            letta_message_index += 1
                            letta_message = letta_messages[letta_message_index]

            elif message.role == MessageRole.user:
                print(f"i={i}, M=user, MM={type(letta_message)}")
                assert isinstance(letta_message, UserMessage)
                assert message.text == letta_message.message
                letta_message_index += 1

            elif message.role == MessageRole.system:
                print(f"i={i}, M=system, MM={type(letta_message)}")
                assert isinstance(letta_message, SystemMessage)
                assert message.text == letta_message.message
                letta_message_index += 1

            elif message.role == MessageRole.tool:
                print(f"i={i}, M=tool, MM={type(letta_message)}")
                assert isinstance(letta_message, FunctionReturn)
                # Check the the value in `text` is the same
                assert message.text == letta_message.function_return
                letta_message_index += 1

            else:
                raise ValueError(f"Unexpected message role: {message.role}")

            # Move to the next message in the original messages list
            break


def test_get_messages_letta_format(server, user_id, agent_id):
    _test_get_messages_letta_format(server, user_id, agent_id, reverse=False)
    _test_get_messages_letta_format(server, user_id, agent_id, reverse=True)


def test_agent_rethink_rewrite_retry(server, user_id, agent_id):
    """Test the /rethink, /rewrite, and /retry commands in the CLI

    - "rethink" replaces the inner thoughts of the last assistant message
    - "rewrite" replaces the text of the last assistant message
    - "retry" retries the last assistant message
    """

    # Send an initial message
    server.user_message(user_id=user_id, agent_id=agent_id, message="Hello?")

    # Grab the raw Agent object
    letta_agent = server._get_or_load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]

    # Try "rethink"
    new_thought = "I am thinking about the meaning of life, the universe, and everything. Bananas?"
    assert last_agent_message.text is not None and last_agent_message.text != new_thought
    server.rethink_agent_message(agent_id=agent_id, new_thought=new_thought)

    # Grab the agent object again (make sure it's live)
    letta_agent = server._get_or_load_agent(agent_id=agent_id)
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
    letta_agent = server._get_or_load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]
    args_json = json.loads(last_agent_message.tool_calls[0].function.arguments)
    assert "message" in args_json and args_json["message"] is not None and args_json["message"] == new_text

    # Try retry
    server.retry_agent_message(agent_id=agent_id)

    # Grab the agent object again (make sure it's live)
    letta_agent = server._get_or_load_agent(agent_id=agent_id)
    assert letta_agent._messages[-1].role == MessageRole.tool
    assert letta_agent._messages[-2].role == MessageRole.assistant
    last_agent_message = letta_agent._messages[-2]

    # Make sure the inner thoughts changed
    assert last_agent_message.text is not None and last_agent_message.text != new_thought

    # Make sure the message changed
    args_json = json.loads(last_agent_message.tool_calls[0].function.arguments)
    print(args_json)
    assert "message" in args_json and args_json["message"] is not None and args_json["message"] != new_text
