import asyncio
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class UserMessage(BaseModel):
    user_id: str
    agent_id: str
    message: str
    stream: bool = False


def setup_agents_message_router(server: SyncServer, interface: QueuingInterface):
    @router.post("/agents/message")
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

    return router
