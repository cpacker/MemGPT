import pytest

from memgpt import create_client


@pytest.fixture(scope="module")
def client():
    yield create_client()


def test_tools(client):

    def print_tool(message: str):
        """
        A tool to print a message

        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.

        """
        print(message)
        return message

    def print_tool2(msg: str):
        """
        Another tool to print a message

        Args:
            msg (str): The message to print.
        """
        print(msg)

    # create tool
    orig_tool_length = len(client.list_tools())
    tool = client.create_tool(print_tool, tags=["extras"])

    # list tools
    tools = client.list_tools()
    assert tool.name in [t.name for t in tools]

    # get tool id
    assert tool.id == client.get_tool_id(name="print_tool")

    # update tool: extras
    extras2 = ["extras2"]
    client.update_tool(tool.id, tags=extras2)
    assert client.get_tool(tool.id).tags == extras2

    # update tool: source code
    client.update_tool(tool.id, name="print_tool2", func=print_tool2)
    assert client.get_tool(tool.id).name == "print_tool2"

    # delete tool
    client.delete_tool(tool.id)
    assert len(client.list_tools()) == orig_tool_length
