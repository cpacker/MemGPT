from pathlib import Path
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio

from letta.settings import settings
from installable_image import InstallableImage
from installable_logger import get_logger

logger = get_logger(__name__)

target_file = settings.letta_dir / "logs" / "letta.log"

app = FastAPI()


async def log_reader(n=5):
    log_lines = []
    with open(target_file, "r") as file:
        for line in file.readlines()[-n:]:
            if line is None:
                continue
            if line.__contains__("ERROR"):
                log_line = {"content": line, "color": "red"}
            elif line.__contains__("WARNING"):
                log_line = {"content": line, "color": "yellow"}
            else:
                log_line = {"content": line, "color": "green"}
            log_lines.append(log_line)
        return log_lines

@app.get("/log")
async def rest_endpoint_log():
    """used to debug log_reader on the fly"""
    logs = await log_reader(30)
    return JSONResponse(logs)

@app.websocket("/ws/log")
async def websocket_endpoint_log(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(1)
            logs = await log_reader(30)
            await websocket.send_json(logs)
    except Exception as e:
        logger.error(f"Error in log websocket: {e}")

    finally:
        await websocket.close()

@app.get("/")
async def get(request: Request):
    context = {"log_file": target_file, "icon": InstallableImage().get_icon_path()}
    templates = Jinja2Templates(directory=str((Path(__file__).parent / "templates").absolute()))
    return templates.TemplateResponse("index.html", {"request": request, "context": context})
