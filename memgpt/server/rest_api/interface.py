from memgpt.interface import AgentInterface


class QueuingInterface(AgentInterface):
    """Messages are queued inside an internal buffer and manually flushed"""

    def __init__(self):
        self.buffer = []

    def clear(self):
        self.buffer = []

    def user_message(self, msg: str):
        """Handle reception of a user message"""
        pass

    def internal_monologue(self, msg: str) -> None:
        """Handle the agent's internal monologue"""
        print(msg)
        self.buffer.append({"internal_monologue": msg})

    def assistant_message(self, msg: str) -> None:
        """Handle the agent sending a message"""
        print(msg)
        self.buffer.append({"assistant_message": msg})

    def function_message(self, msg: str) -> None:
        """Handle the agent calling a function"""
        print(msg)

        if msg.startswith("Running "):
            msg = msg.replace("Running ", "")
            self.buffer.append({"function_call": msg})

        elif msg.startswith("Success: "):
            msg = msg.replace("Success: ", "")
            self.buffer.append({"function_return": msg, "status": "success"})

        elif msg.startswith("Error: "):
            msg = msg.replace("Error: ", "")
            self.buffer.append({"function_return": msg, "status": "error"})

        else:
            # NOTE: generic, should not happen
            self.buffer.append({"function_message": msg})
