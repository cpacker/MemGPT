import asyncio
import queue

from memgpt.interface import AgentInterface


class QueuingInterface(AgentInterface):
    """Messages are queued inside an internal buffer and manually flushed"""

    def __init__(self):
        self.buffer = queue.Queue()

    def to_list(self):
        """Convert queue to a list (empties it out at the same time)"""
        items = []
        while not self.buffer.empty():
            try:
                items.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        if len(items) > 1 and items[-1] == "STOP":
            items.pop()
        return items

    def clear(self):
        """Clear all messages from the queue."""
        with self.buffer.mutex:
            # Empty the queue
            self.buffer.queue.clear()

    async def message_generator(self):
        while True:
            if not self.buffer.empty():
                message = self.buffer.get()
                if message == "STOP":
                    break
                yield message
            else:
                await asyncio.sleep(0.1)  # Small sleep to prevent a busy loop

    def step_yield(self):
        """Enqueue a special stop message"""
        self.buffer.put("STOP")

    def user_message(self, msg: str):
        """Handle reception of a user message"""
        pass

    def internal_monologue(self, msg: str) -> None:
        """Handle the agent's internal monologue"""
        print(msg)
        self.buffer.put({"internal_monologue": msg})

    def assistant_message(self, msg: str) -> None:
        """Handle the agent sending a message"""
        print(msg)
        self.buffer.put({"assistant_message": msg})

    def function_message(self, msg: str) -> None:
        """Handle the agent calling a function"""
        print(msg)

        if msg.startswith("Running "):
            msg = msg.replace("Running ", "")
            self.buffer.put({"function_call": msg})

        elif msg.startswith("Success: "):
            msg = msg.replace("Success: ", "")
            self.buffer.put({"function_return": msg, "status": "success"})

        elif msg.startswith("Error: "):
            msg = msg.replace("Error: ", "")
            self.buffer.put({"function_return": msg, "status": "error"})

        else:
            # NOTE: generic, should not happen
            self.buffer.put({"function_message": msg})
