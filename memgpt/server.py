
# server.py
import asyncio
import websockets

connected_clients = set()
messages_queue = asyncio.Queue()

async def server(websocket, path):
    # Register client connection
    connected_clients.add(websocket)
    print(connected_clients)
    print("connected_clients rocking")
    try:
        while True:
            # Receive and process message
            message = await websocket.recv()
            print(f"Received message: {message}")
            
            # You might want to add received messages to the queue here
            await messages_queue.put(message)
            print(messages_queue)
            
            # Here, we're broadcasting messages to all connected clients
            for client in connected_clients:
                if client != websocket:  # Avoid sending the message back to the sender
                    await client.send(message)
    except websockets.exceptions.ConnectionClosed:
        # Unregister clients and handle connection closure
        connected_clients.remove(websocket)

start_server = websockets.serve(server, "localhost", 1234)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
