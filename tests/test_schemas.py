from memgpt.schemas.block import Block
from memgpt.schemas.message import Message


def test_block():

    # throw error on creation
    threw_error = False
    try:
        block = Block(value="x" * 50, limit=10)
    except ValueError:
        threw_error = True
    assert threw_error, "Should have thrown an error when value exceeds limit"

    block = Block(value="x" * 50, limit=100)

    # throw error when value exceeds limit
    threw_error = False
    try:
        block.value = "x" * 101
    except ValueError:
        threw_error = True
    assert threw_error, "Should have thrown an error when setting value to exceed limit"

    # thow error if limit is less than value
    threw_error = False
    try:
        block.limit = 10
    except ValueError:
        threw_error = True
    assert threw_error, "Should have thrown an error when setting limit to be less than value"


def test_message():
    user_message = Message(text="Hello", role="user")

    tool_message = Message(text="Hello", role="tool")
