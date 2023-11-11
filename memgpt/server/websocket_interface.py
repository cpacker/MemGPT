import asyncio
import threading


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
    def __init__(self):
        super().__init__()
        self.clients = set()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop)
        self.thread.start()

    def _run_event_loop(self):
        """Run the dedicated event loop and handle its closure."""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            # Run the cleanup tasks in the event loop
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def _run_async(self, coroutine):
        """Schedule coroutine to be run in the dedicated event loop."""
        if not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _send_to_all_clients(self, clients, msg):
        """Asynchronously sends a message to all clients."""
        if clients:
            await asyncio.gather(*(client.send(msg) for client in clients))

    def user_message(self, msg):
        """Handle reception of a user message"""
        # Logic to process the user message and possibly trigger agent's response
        pass

    def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, f"Internal monologue: {msg}"))

    def assistant_message(self, msg):
        """Handle the agent sending a message"""
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, f"Assistant message: {msg}"))

    def function_message(self, msg):
        """Handle the agent calling a function"""
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, f"Function call: {msg}"))

    def close(self):
        """Shut down the WebSocket interface and its event loop."""
        self.loop.call_soon_threadsafe(self.loop.stop)  # Signal the loop to stop
        self.thread.join()  # Wait for the thread to finish
