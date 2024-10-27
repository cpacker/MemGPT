import pytest

from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.schemas.tool import Tool
from letta.services.secure_execution_environment import SecureExecutionEnvironment


def test_function_with_no_params():

    def print_hello_world():
        """
        Simple function that prints hello world

        Returns:
            str: The string hello world.

        """
        print("hello world")
        return "hello world"

    tool = create_tool_from_func(print_hello_world)
    args = {}

    response = SecureExecutionEnvironment(tool, args).run()
    assert response == "hello world"


def test_function_with_no_rv():

    def print_hello_world():
        """Simple function that prints hello world"""
        
        print("hello world")

    tool = create_tool_from_func(print_hello_world)
    args = {}

    response = SecureExecutionEnvironment(tool, args).run()
    assert response == None


def test_function_with_list_rv():

    def create_list():
        """Simple function that returns a list"""
        
        return [1] * 5

    tool = create_tool_from_func(create_list)
    args = {}

    response = SecureExecutionEnvironment(tool, args).run()
    assert response == [1, 1, 1, 1, 1]


def test_function_with_single_param():

    def print_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.

        """
        print(message)
        return message

    tool = create_tool_from_func(print_tool)
    args = {"message": "hey hi hello"}

    response = SecureExecutionEnvironment(tool, args).run()
    assert response == args["message"]


def test_function_with_exception():

    def throw_exception(message: str):
        """
        Args:
            message (str): The exception message to throw.

        """
        raise Exception(message)


    tool = create_tool_from_func(throw_exception)
    args = {"message": "major silliness"}

    with pytest.raises(Exception, match=args["message"]):
        response = SecureExecutionEnvironment(tool, args).run()


def test_function_with_missing_param():

    def print_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.

        """
        print(message)
        return message

    tool = create_tool_from_func(print_tool)
    args = {}

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'message'"):
        SecureExecutionEnvironment(tool, args).run()

def test_function_with_extra_param():

    def print_tool(message: str):
        """
        Args:
            message (str): The message to print.

        Returns:
            str: The message that was printed.

        """
        print(message)
        return message

    tool = create_tool_from_func(print_tool)
    args = {"message": "hey hi hello", "self": tool}

    response = SecureExecutionEnvironment(tool, args).run()
    assert response == args["message"]


def test_function_with_multi_param():

    def multi_print_tool(message: str, times: int, explode: bool):
        """
        Args:
            message (str): The message to print.
            times (int): How many times to print.
            explode (bool): Whether message should be reversed.

        Returns:
            str: The message that was printed.

        """
        message = message * times
        if explode:
            message = " ".join([char for char in message])
        print(message)
        return message

    tool = create_tool_from_func(multi_print_tool)
    args = {
        "message": "meow",
        "times": 3,
        "explode": True,
    }

    response = SecureExecutionEnvironment(tool, args).run()
    assert response == "m e o w m e o w m e o w"


def test_function_with_object_param():

    def print_tool_name(message: str, tool: Tool):
        """
        Args:
            message (str): The message to print.
            tool (Tool): The tool to print the name of.

        Returns:
            str: The message that was printed.

        """
        print(message + ": " + tool.name)
        return message

    tool = create_tool_from_func(print_tool_name)
    args = {
        "message": "This is a cool tool",
        "tool": tool,
    }

    with pytest.raises(TypeError, match="unsupported type: object"):
        SecureExecutionEnvironment(tool, args).run()


def create_tool_from_func(func: callable):
    return Tool(
        name=func.__name__,
        description="",
        source_type="python",
        tags=[],
        source_code=parse_source_code(func),
        json_schema=generate_schema(func, None),
    )