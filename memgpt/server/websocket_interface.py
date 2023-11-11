import asyncio

from ..interface import AgentInterface


class BaseWebSocketInterface(AgentInterface):
    """Interface for interacting with a MemGPT agent over a WebSocket"""

    def __init__(self):
        self.clients = set()

    def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)

    def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.remove(websocket)


class AsyncWebSocketInterface(BaseWebSocketInterface):
    """WebSocket calls are async"""

    async def user_message(self, msg):
        """Handle reception of a user message"""
        # Logic to process the user message and possibly trigger agent's response
        pass

    async def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        # Send the internal monologue to all clients
        if self.clients:  # Check if there are any clients connected
            await asyncio.gather(*[client.send(f"Internal monologue: {msg}") for client in self.clients])

    async def assistant_message(self, msg):
        """Handle the agent sending a message"""
        # Send the assistant's message to all clients
        if self.clients:
            await asyncio.gather(*[client.send(f"Assistant message: {msg}") for client in self.clients])

    async def function_message(self, msg):
        """Handle the agent calling a function"""
        # Send the function call message to all clients
        if self.clients:
            await asyncio.gather(*[client.send(f"Function call: {msg}") for client in self.clients])


class BlockingWebSocketInterface(BaseWebSocketInterface):
    """No async signatures, calls are blocking instead"""

    def user_message(self, msg):
        """Handle reception of a user message"""
        # Logic to process the user message and possibly trigger agent's response
        pass

    def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        if self.clients:
            loop = asyncio.get_event_loop()
            tasks = [loop.create_task(client.send(f"Internal monologue: {msg}")) for client in self.clients]
            loop.run_until_complete(asyncio.gather(*tasks))

    def assistant_message(self, msg):
        """Handle the agent sending a message"""
        if self.clients:
            loop = asyncio.get_event_loop()
            tasks = [loop.create_task(client.send(f"Assistant message: {msg}")) for client in self.clients]
            loop.run_until_complete(asyncio.gather(*tasks))

    def function_message(self, msg):
        """Handle the agent calling a function"""
        if self.clients:
            loop = asyncio.get_event_loop()
            tasks = [loop.create_task(client.send(f"Function call: {msg}")) for client in self.clients]
            loop.run_until_complete(asyncio.gather(*tasks))
