import asyncio

from ..interface import AgentInterface


class WebSocketInterface(AgentInterface):
    """Interface for interacting with a MemGPT agent over a WebSocket"""

    def __init__(self):
        pass

    async def user_message(self, msg):
        """MemGPT receives a user message"""
        pass

    async def internal_monologue(self, msg):
        """MemGPT generates some internal monologue"""
        pass

    async def assistant_message(self, msg):
        """MemGPT uses send_message"""
        pass

    async def function_message(self, msg):
        """MemGPT calls a function"""
        pass


class WebSocketInterface(AgentInterface):
    """Interface for interacting with a MemGPT agent over a WebSocket"""

    def __init__(self, websocket):
        self.websocket = websocket
        self.clients = set()

    async def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)

    async def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.remove(websocket)

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
