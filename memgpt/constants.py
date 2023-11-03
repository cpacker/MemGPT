import os

MEMGPT_DIR = os.path.join(os.path.expanduser("~"), ".memgpt")

DEFAULT_MEMGPT_MODEL = "gpt-4"

FIRST_MESSAGE_ATTEMPTS = 10

INITIAL_BOOT_MESSAGE = "Boot sequence complete. Persona activated."
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT = "Bootup sequence complete. Persona activated. Testing messaging functionality."
STARTUP_QUOTES = [
    "I think, therefore I am.",
    "All those moments will be lost in time, like tears in rain.",
    "More human than human is our motto.",
]
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG = STARTUP_QUOTES[2]

# Constants to do with summarization / conversation length window
# The max amount of tokens supported by the underlying model (eg 8k for gpt-4 and Mistral 7B)
LLM_MAX_TOKENS = 8000  # change this depending on your model
# The amount of tokens before a sytem warning about upcoming truncation is sent to MemGPT
MESSAGE_SUMMARY_WARNING_TOKENS = int(0.75 * LLM_MAX_TOKENS)
# The error message that MemGPT will receive
MESSAGE_SUMMARY_WARNING_STR = f"Warning: the conversation history will soon reach its maximum length and be trimmed. Make sure to save any important information from the conversation to your memory before it is removed."
# The fraction of tokens we truncate down to
MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC = 0.75

# Even when summarizing, we want to keep a handful of recent messages
# These serve as in-context examples of how to use functions / what user messages look like
MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST = 3

# Default memory limits
CORE_MEMORY_PERSONA_CHAR_LIMIT = 2000
CORE_MEMORY_HUMAN_CHAR_LIMIT = 2000

MAX_PAUSE_HEARTBEATS = 360  # in min

MESSAGE_CHATGPT_FUNCTION_MODEL = "gpt-3.5-turbo"
MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE = "You are a helpful assistant. Keep your responses short and concise."

#### Functions related

REQ_HEARTBEAT_MESSAGE = "request_heartbeat == true"
FUNC_FAILED_HEARTBEAT_MESSAGE = "Function call failed"
FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT = "Request an immediate heartbeat after function execution. Set to 'true' if you want to send a follow-up message or run a follow-up function."
