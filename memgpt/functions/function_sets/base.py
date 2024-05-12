from typing import Optional
import json
import math

from memgpt.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE, JSON_ENSURE_ASCII

### Functions / tools the agent can use
# All functions should return a response string (or None)
# If the function fails, throw an exception


def send_message(self, message: str) -> Optional[str]:
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.assistant_message(message)
    return None


def conversation_search(self, query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search prior conversation history using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.persistence_manager.recall_memory.text_search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def conversation_search_date(self, start_date: str, end_date: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search prior conversation history using a date range.

    Args:
        start_date (str): The start of the date range to search, in the format 'YYYY-MM-DD'.
        end_date (str): The end of the date range to search, in the format 'YYYY-MM-DD'.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.persistence_manager.recall_memory.date_search(start_date, end_date, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def archival_memory_insert(self, content: str) -> Optional[str]:
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.persistence_manager.archival_memory.insert(content)
    return None


def archival_memory_search(self, query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search archival memory using semantic (embedding-based) search.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.persistence_manager.archival_memory.search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, memory: {d['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str
