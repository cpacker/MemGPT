import json

from letta.agent import Agent


def dump_json(self: Agent, input: str) -> str:
    """
    Dumps the content to JSON.

    Args:
        input (dict): dictionary object to convert to a string

    Returns:
        str: returns string version of the input
    """
    return json.dumps(input)
