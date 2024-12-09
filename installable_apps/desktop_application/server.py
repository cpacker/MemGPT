from typing import TYPE_CHECKING
from uvicorn import Server, Config
from time import sleep
import threading
from contextlib import contextmanager

if TYPE_CHECKING:
    from fastapi import WSGIApplication

class ThreadedServer(Server):

    @contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                sleep(0.1)
            yield
        finally:
            self.should_exit = True
            thread.join()

    @classmethod
    def get_configured_server(cls,
                              app: "WSGIApplication",
                              port: int,
                              host: str,
                              log_level: str = "info") -> "ThreadedServer":
        config = Config(app, host=host, port=port, log_level="info")
        return cls(config=config)