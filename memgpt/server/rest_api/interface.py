from memgpt.interface import AgentInterface


class QueuingInterface(AgentInterface):
    """Messages are queued inside an internal buffer and manually flushed"""

    def __init__(self):
        self.buffer = []

    def clear(self):
        self.buffer = []

    def user_message(self, msg):
        """Handle reception of a user message"""
        pass

    def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        print(msg)
        self.buffer.append(msg)

    def assistant_message(self, msg):
        """Handle the agent sending a message"""
        print(msg)
        self.buffer.append(msg)

    def function_message(self, msg):
        """Handle the agent calling a function"""
        print(msg)
        self.buffer.append(msg)
