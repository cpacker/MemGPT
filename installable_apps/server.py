from uvicorn import Server, Config
from time import sleep
import threading
from contextlib import contextmanager
from letta.server.rest_api.app import app
from letta.server.constants import REST_DEFAULT_PORT

class ThreadedServer(Server):
    #def install_signal_handlers(self):
    #    pass

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
    def get_configured_server(cls):
        config = Config(app, host="localhost", port=REST_DEFAULT_PORT, log_level="info")
        return cls(config=config)