import asyncio
from contextlib import asynccontextmanager
import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface

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


class CoreMemory(BaseModel):
    user_id: str
    agent_id: str
    human: str | None = None
    persona: str | None = None


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


server: SyncServer | None = None
interface: QueuingInterface | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global server
    global interface
    interface = QueuingInterface()
    server = SyncServer(default_interface=interface)
    yield
    server.save_agents()
    server = None


CORS_ORIGINS = [
    "http://localhost:4200",
    "http://localhost:4201",
    "http://localhost:8283",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:4201",
    "http://127.0.0.1:8283",
]

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents")
def list_agents(user_id: str):
    interface.clear()
    return server.list_agents(user_id=user_id)


@app.get("/agents/memory")
def get_agent_memory(user_id: str, agent_id: str):
    interface.clear()
    return server.get_agent_memory(user_id=user_id, agent_id=agent_id)


@app.put("/agents/memory")
def get_agent_memory(body: CoreMemory):
    interface.clear()
    new_memory_contents = {"persona": body.persona, "human": body.human}
    return server.update_agent_core_memory(user_id=body.user_id, agent_id=body.agent_id, new_memory_contents=new_memory_contents)


@app.get("/agents/config")
def get_agent_config(user_id: str, agent_id: str):
    interface.clear()
    return server.get_agent_config(user_id=user_id, agent_id=agent_id)


@app.get("/config")
def get_server_config(user_id: str):
    interface.clear()
    return server.get_server_config(user_id=user_id)


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
                await asyncio.create_task(server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message))
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
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    else:
        interface.clear()
        try:
            server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return {"messages": interface.to_list()}


# server.run_command
@app.post("/agents/command")
def run_command(body: Command):
    interface.clear()
    try:
        response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return {"response": response}


app.mount(
    "/",
    SPAStaticFiles(
        directory=os.path.join(os.getcwd(), "..", "static_files"),
        html=True,
    ),
    name="spa-static-files",
)
