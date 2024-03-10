from ..constants import FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT, MAX_PAUSE_HEARTBEATS

# FUNCTIONS_PROMPT_MULTISTEP_NO_HEARTBEATS = FUNCTIONS_PROMPT_MULTISTEP[:-1]
FUNCTIONS_CHAINING = {
    "send_message": {
        "name": "send_message",
        "description": "Sends a message to the human user.",
        "parameters": {
            "type": "object",
            "properties": {
                # https://json-schema.org/understanding-json-schema/reference/array.html
                "message": {
                    "type": "string",
                    "description": "Message contents. All unicode (including emojis) are supported.",
                },
            },
            "required": ["message"],
        },
    },
    "pause_heartbeats": {
        "name": "pause_heartbeats",
        "description": "Temporarily ignore timed heartbeats. You may still receive messages from manual heartbeats and other events.",
        "parameters": {
            "type": "object",
            "properties": {
                # https://json-schema.org/understanding-json-schema/reference/array.html
                "minutes": {
                    "type": "integer",
                    "description": f"Number of minutes to ignore heartbeats for. Max value of {MAX_PAUSE_HEARTBEATS} minutes ({MAX_PAUSE_HEARTBEATS//60} hours).",
                },
            },
            "required": ["minutes"],
        },
    },
    "message_chatgpt": {
        "name": "message_chatgpt",
        "description": "Send a message to a more basic AI, ChatGPT. A useful resource for asking questions. ChatGPT does not retain memory of previous interactions.",
        "parameters": {
            "type": "object",
            "properties": {
                # https://json-schema.org/understanding-json-schema/reference/array.html
                "message": {
                    "type": "string",
                    "description": "Message to send ChatGPT. Phrase your message as a full English sentence.",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["message", "request_heartbeat"],
        },
    },
    "core_memory_append": {
        "name": "core_memory_append",
        "description": "Append to the contents of core memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Section of the memory to be edited (persona or human).",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the memory. All unicode (including emojis) are supported.",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["name", "content", "request_heartbeat"],
        },
    },
    "core_memory_replace": {
        "name": "core_memory_replace",
        "description": "Replace the contents of core memory. To delete memories, use an empty string for new_content.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Section of the memory to be edited (persona or human).",
                },
                "old_content": {
                    "type": "string",
                    "description": "String to replace. Must be an exact match.",
                },
                "new_content": {
                    "type": "string",
                    "description": "Content to write to the memory. All unicode (including emojis) are supported.",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["name", "old_content", "new_content", "request_heartbeat"],
        },
    },
    "recall_memory_search": {
        "name": "recall_memory_search",
        "description": "Search prior conversation history using a string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String to search for.",
                },
                "page": {
                    "type": "integer",
                    "description": "Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["query", "page", "request_heartbeat"],
        },
    },
    "conversation_search": {
        "name": "conversation_search",
        "description": "Search prior conversation history using case-insensitive string matching.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String to search for.",
                },
                "page": {
                    "type": "integer",
                    "description": "Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["query", "request_heartbeat"],
        },
    },
    "recall_memory_search_date": {
        "name": "recall_memory_search_date",
        "description": "Search prior conversation history using a date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "The start of the date range to search, in the format 'YYYY-MM-DD'.",
                },
                "end_date": {
                    "type": "string",
                    "description": "The end of the date range to search, in the format 'YYYY-MM-DD'.",
                },
                "page": {
                    "type": "integer",
                    "description": "Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["start_date", "end_date", "page", "request_heartbeat"],
        },
    },
    "conversation_search_date": {
        "name": "conversation_search_date",
        "description": "Search prior conversation history using a date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "The start of the date range to search, in the format 'YYYY-MM-DD'.",
                },
                "end_date": {
                    "type": "string",
                    "description": "The end of the date range to search, in the format 'YYYY-MM-DD'.",
                },
                "page": {
                    "type": "integer",
                    "description": "Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["start_date", "end_date", "request_heartbeat"],
        },
    },
    "archival_memory_insert": {
        "name": "archival_memory_insert",
        "description": "Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to write to the memory. All unicode (including emojis) are supported.",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["content", "request_heartbeat"],
        },
    },
    "archival_memory_search": {
        "name": "archival_memory_search",
        "description": "Search archival memory using semantic (embedding-based) search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String to search for.",
                },
                "page": {
                    "type": "integer",
                    "description": "Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["query", "request_heartbeat"],
        },
    },
    "read_from_text_file": {
        "name": "read_from_text_file",
        "description": "Read lines from a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to read.",
                },
                "line_start": {
                    "type": "integer",
                    "description": "Line to start reading from.",
                },
                "num_lines": {
                    "type": "integer",
                    "description": "How many lines to read (defaults to 1).",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["filename", "line_start", "request_heartbeat"],
        },
    },
    "append_to_text_file": {
        "name": "append_to_text_file",
        "description": "Append to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to append to.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to append to the file.",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["filename", "content", "request_heartbeat"],
        },
    },
    "http_request": {
        "name": "http_request",
        "description": "Generates an HTTP request and returns the response.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "The HTTP method (e.g., 'GET', 'POST').",
                },
                "url": {
                    "type": "string",
                    "description": "The URL for the request.",
                },
                "payload_json": {
                    "type": "string",
                    "description": "A JSON string representing the request payload.",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT,
                },
            },
            "required": ["method", "url", "request_heartbeat"],
        },
    },
}
