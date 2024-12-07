from datetime import datetime
from typing import Optional

from letta.agent import Agent
from letta.constants import MAX_PAUSE_HEARTBEATS

# import math
# from letta.utils import json_dumps

### Functions / tools the agent can use
# All functions should return a response string (or None)
# If the function fails, throw an exception


def send_message(self: "Agent", message: str) -> Optional[str]:
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # FIXME passing of msg_obj here is a hack, unclear if guaranteed to be the correct reference
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    return None


# Construct the docstring dynamically (since it should use the external constants)
pause_heartbeats_docstring = f"""
Temporarily ignore timed heartbeats. You may still receive messages from manual heartbeats and other events.

Args:
    minutes (int): Number of minutes to ignore heartbeats for. Max value of {MAX_PAUSE_HEARTBEATS} minutes ({MAX_PAUSE_HEARTBEATS // 60} hours).

Returns:
    str: Function status response
"""


def pause_heartbeats(self: "Agent", minutes: int) -> Optional[str]:
    import datetime

    from letta.constants import MAX_PAUSE_HEARTBEATS

    minutes = min(MAX_PAUSE_HEARTBEATS, minutes)

    # Record the current time
    self.pause_heartbeats_start = datetime.datetime.now(datetime.timezone.utc)
    # And record how long the pause should go for
    self.pause_heartbeats_minutes = int(minutes)

    return f"Pausing timed heartbeats for {minutes} min"


pause_heartbeats.__doc__ = pause_heartbeats_docstring


def conversation_search(self: "Agent", query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search prior conversation history using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """

    import math

    from letta.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    from letta.utils import json_dumps

    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    # TODO: add paging by page number. currently cursor only works with strings.
    # original: start=page * count
    results = self.message_manager.list_user_messages_for_agent(
        agent_id=self.agent_state.id,
        actor=self.user,
        query_text=query,
        limit=count,
    )
    total = len(results)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str


def conversation_search_date(self: "Agent", start_date: str, end_date: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search prior conversation history using a date range.

    Args:
        start_date (str): The start of the date range to search, in the format 'YYYY-MM-DD'.
        end_date (str): The end of the date range to search, in the format 'YYYY-MM-DD'.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    import math

    from letta.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    from letta.utils import json_dumps

    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
        if page < 0:
            raise ValueError
    except:
        raise ValueError(f"'page' argument must be an integer")

    # Convert date strings to datetime objects
    try:
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
    except ValueError:
        raise ValueError("Dates must be in the format 'YYYY-MM-DD'")

    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results = self.message_manager.list_user_messages_for_agent(
        # TODO: add paging by page number. currently cursor only works with strings.
        agent_id=self.agent_state.id,
        actor=self.user,
        start_date=start_datetime,
        end_date=end_datetime,
        limit=count,
        # start_date=start_date, end_date=end_date, limit=count, start=page * count
    )
    total = len(results)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str


def archival_memory_insert(self: "Agent", content: str) -> Optional[str]:
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.archival_memory.insert(content)
    return None


def archival_memory_search(self: "Agent", query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search archival memory using semantic (embedding-based) search.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    import math

    from letta.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    from letta.utils import json_dumps

    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.archival_memory.search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, memory: {d['content']}" for d in results]
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str


def core_memory_append(agent_state: "AgentState", label: str, content: str) -> Optional[str]:  # type: ignore
    """
    Append to the contents of core memory.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    new_value = current_value + "\n" + str(content)
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None


def core_memory_replace(agent_state: "AgentState", label: str, old_content: str, new_content: str) -> Optional[str]:  # type: ignore
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    if old_content not in current_value:
        raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
    new_value = current_value.replace(str(old_content), str(new_content))
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None
