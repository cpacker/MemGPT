from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends

# Initialize FastAPI application
web_app = FastAPI()

# Configure CORS settings to allow all origins, methods, and headers.
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class MemGPTAgent:
    def __init__(self):
        self.agent = None

    def set_agent(self, agent_obj):
        self.agent = agent_obj

    async def handle_message(self, user_message: str):
        if self.agent is not None:
            new_messages, _, _, _ = await self.agent.step(user_message, first_message=True)
            for message in new_messages:
                if 'content' in message:
                    return message['content']
                if 'function_call' in message and message['function_call']['name'] == 'send_message':
                    function_message = json.loads(message['function_call']['arguments'])['message']
                    return function_message
        else:
            print("memgpt_agent is not set")
            return None

agent_manager = MemGPTAgent()

async def get_agent_manager():
    return agent_manager

@web_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, agent_manager: MemGPTAgent = Depends(get_agent_manager)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                print("Received empty message, ignoring.")
                continue

            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
            except json.JSONDecodeError:
                print("Invalid JSON received:", data)
                await websocket.send_text(json.dumps({"error": "Invalid JSON received"}))
                continue

            reply = await agent_manager.handle_message(user_message)
            if reply is not None:
                await websocket.send_text(reply)

    except WebSocketDisconnect:
        print("Client disconnected")
