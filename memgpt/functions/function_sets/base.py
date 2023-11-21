from typing import Optional
import datetime
import os
import json
import math

from memgpt.constants import MAX_PAUSE_HEARTBEATS, RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

### Functions / tools the agent can use
# All functions should return a response string (or None)
# If the function fails, throw an exception


def send_message(self, message: str):
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.assistant_message(message)
    return None


# Construct the docstring dynamically (since it should use the external constants)
pause_heartbeats_docstring = f"""
Temporarily ignore timed heartbeats. You may still receive messages from manual heartbeats and other events.

Args:
    minutes (int): Number of minutes to ignore heartbeats for. Max value of {MAX_PAUSE_HEARTBEATS} minutes ({MAX_PAUSE_HEARTBEATS // 60} hours).

Returns:
    str: Function status response
"""


def pause_heartbeats(self, minutes: int):
    minutes = min(MAX_PAUSE_HEARTBEATS, minutes)

    # Record the current time
    self.pause_heartbeats_start = datetime.datetime.now()
    # And record how long the pause should go for
    self.pause_heartbeats_minutes = int(minutes)

    return f"Pausing timed heartbeats for {minutes} min"


pause_heartbeats.__doc__ = pause_heartbeats_docstring


def core_memory_append(self, name: str, content: str):
    """
    Append to the contents of core memory.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    new_len = self.memory.edit_append(name, content)
    self.rebuild_memory()
    return None


def core_memory_replace(self, name: str, old_content: str, new_content: str):
    """
    Replace to the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    new_len = self.memory.edit_replace(name, old_content, new_content)
    self.rebuild_memory()
    return None


def conversation_search(self, query: str, page: Optional[int] = 0):
    """
    Search prior conversation history using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.persistence_manager.recall_memory.text_search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted)}"
    return results_str


def conversation_search_date(self, start_date: str, end_date: str, page: Optional[int] = 0):
    """
    Search prior conversation history using a date range.

    Args:
        start_date (str): The start of the date range to search, in the format 'YYYY-MM-DD'.
        end_date (str): The end of the date range to search, in the format 'YYYY-MM-DD'.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.persistence_manager.recall_memory.date_search(start_date, end_date, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted)}"
    return results_str


def archival_memory_insert(self, content: str):
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.persistence_manager.archival_memory.insert(content)
    return None


def archival_memory_search(self, query: str, page: Optional[int] = 0):
    """
    Search archival memory using semantic (embedding-based) search.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.persistence_manager.archival_memory.search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, memory: {d['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted)}"
    return results_str
