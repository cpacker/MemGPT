import json
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

# Avoid circular imports
if TYPE_CHECKING:
    from letta.schemas.message import Message


class ErrorCode(Enum):
    """Enum for error codes used by client."""

    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"


class LettaError(Exception):
    """Base class for all Letta related errors."""

    def __init__(self, message: str, code: Optional[ErrorCode] = None, details: dict = {}):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.code:
            return f"{self.code.value}: {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', code='{self.code}', details={self.details})"


class LettaToolCreateError(LettaError):
    """Error raised when a tool cannot be created."""

    default_error_message = "Error creating tool."

    def __init__(self, message=None):
        super().__init__(message=message or self.default_error_message)


class LettaConfigurationError(LettaError):
    """Error raised when there are configuration-related issues."""

    def __init__(self, message: str, missing_fields: Optional[List[str]] = None):
        self.missing_fields = missing_fields or []
        super().__init__(message=message, details={"missing_fields": self.missing_fields})


class LettaAgentNotFoundError(LettaError):
    """Error raised when an agent is not found."""
    pass


class LettaUserNotFoundError(LettaError):
    """Error raised when a user is not found."""
    pass


class LLMError(LettaError):
    pass


class LLMJSONParsingError(LettaError):
    """Exception raised for errors in the JSON parsing process."""

    def __init__(self, message="Error parsing JSON generated by LLM"):
        super().__init__(message=message)


class LocalLLMError(LettaError):
    """Generic catch-all error for local LLM problems"""

    def __init__(self, message="Encountered an error while running local LLM"):
        super().__init__(message=message)


class LocalLLMConnectionError(LettaError):
    """Error for when local LLM cannot be reached with provided IP/port"""

    def __init__(self, message="Could not connect to local LLM"):
        super().__init__(message=message)


class ContextWindowExceededError(LettaError):
    """Error raised when the context window is exceeded but further summarization fails."""

    def __init__(self, message: str, details: dict = {}):
        error_message = f"{message} ({details})"
        super().__init__(
            message=error_message,
            code=ErrorCode.CONTEXT_WINDOW_EXCEEDED,
            details=details,
        )


class LettaMessageError(LettaError):
    """Base error class for handling message-related errors."""

    messages: List[Union["Message", "LettaMessage"]]
    default_error_message: str = "An error occurred with the message."

    def __init__(self, *, messages: List[Union["Message", "LettaMessage"]], explanation: Optional[str] = None) -> None:
        error_msg = self.construct_error_message(messages, self.default_error_message, explanation)
        super().__init__(error_msg)
        self.messages = messages

    @staticmethod
    def construct_error_message(messages: List[Union["Message", "LettaMessage"]], error_msg: str, explanation: Optional[str] = None) -> str:
        """Helper method to construct a clean and formatted error message."""
        if explanation:
            error_msg += f" (Explanation: {explanation})"

        # Pretty print out message JSON
        message_json = json.dumps([message.model_dump() for message in messages], indent=4)
        return f"{error_msg}\n\n{message_json}"


class MissingFunctionCallError(LettaMessageError):
    """Error raised when a message is missing a function call."""

    default_error_message = "The message is missing a function call."


class InvalidFunctionCallError(LettaMessageError):
    """Error raised when a message uses an invalid function call."""

    default_error_message = "The message uses an invalid function call or has improper usage of a function call."


class MissingInnerMonologueError(LettaMessageError):
    """Error raised when a message is missing an inner monologue."""

    default_error_message = "The message is missing an inner monologue."


class InvalidInnerMonologueError(LettaMessageError):
    """Error raised when a message has a malformed inner monologue."""

    default_error_message = "The message has a malformed inner monologue."
