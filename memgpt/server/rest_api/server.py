from typing import Union

from fastapi import FastAPI
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


class CreateConfig(BaseModel):
    user_id: str
    config: dict


class UserMessage(BaseModel):
    user_id: str
    agent_id: str
    message: str


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
    agents_list = utils.list_agent_config_files()
    interface.clear()
    return {"num_agents": len(agents_list), "agent_names": agents_list}


# server.create_agent
@app.post("/agents")
def create_agents(body: CreateConfig):
    agent_id = server.create_agent(user_id=body.user_id, config=body.config)
    interface.clear()
    return {"agent_id": agent_id}


# server.user_message
@app.post("/agents/message")
def user_message(body: UserMessage):
    server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message)
    interface.clear()
    return {"message": interface.buffer}


# server.run_command
@app.post("/agents/command")
def run_command(body: Command):
    response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
    interface.clear()
    return {"response": response}
