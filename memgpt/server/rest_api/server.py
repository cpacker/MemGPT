import asyncio
import json
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
import memgpt.utils as utils


"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""


class CreateAgentConfig(BaseModel):
    user_id: str
    config: dict


class UserMessage(BaseModel):
    user_id: str
    agent_id: str
    message: str
    stream: bool = False


class Command(BaseModel):
    user_id: str
    agent_id: str
    command: str


app = FastAPI()
interface = QueuingInterface()
server = SyncServer(default_interface=interface)


# server.list_agents
@app.get("/agents")
def list_agents(user_id: str):
    interface.clear()
    agents_list = utils.list_agent_config_files()
    return {"num_agents": len(agents_list), "agent_names": agents_list}


# server.create_agent
@app.post("/agents")
def create_agents(body: CreateAgentConfig):
    interface.clear()
    try:
        agent_id = server.create_agent(user_id=body.user_id, agent_config=body.config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return {"agent_id": agent_id}


# server.user_message
@app.post("/agents/message")
async def user_message(body: UserMessage):
    if body.stream:
        # For streaming response
        try:
            # Start the generation process (similar to the non-streaming case)
            # This should be a non-blocking call or run in a background task

            # Check if server.user_message is an async function
            if asyncio.iscoroutinefunction(server.user_message):
                # Start the async task
                asyncio.create_task(server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message))
            else:
                # Run the synchronous function in a thread pool
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, server.user_message, body.user_id, body.agent_id, body.message)

            async def formatted_message_generator():
                async for message in interface.message_generator():
                    formatted_message = f"data: {json.dumps(message)}\n\n"
                    yield formatted_message
                    await asyncio.sleep(1)

            # Return the streaming response using the generator
            return StreamingResponse(formatted_message_generator(), media_type="text/event-stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    else:
        interface.clear()
        try:
            server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return {"messages": interface.to_list()}


# server.run_command
@app.post("/agents/command")
def run_command(body: Command):
    interface.clear()
    try:
        response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
    return {"response": response}
