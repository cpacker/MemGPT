#!/usr/bin/env python3
import pgserver
import webbrowser

from letta.settings import settings
from letta.server.rest_api.app import app as letta_app
from letta.server.constants import REST_DEFAULT_PORT

from server import ThreadedServer
from logserver.main import app as log_app
from tray import Tray

pgdata = settings.letta_dir / "pgdata"
pgdata.mkdir(parents=True, exist_ok=True)

database = pgserver.get_server(pgdata)
# create pg vector extension
#database.psql('CREATE EXTENSION IF NOT EXISTS pg_vector;')

# feed database URI parts to the application
settings.pg_uri = database.get_uri()
# start the servers

app_server = ThreadedServer.get_configured_server(letta_app, host="localhost", port=REST_DEFAULT_PORT)
log_server = ThreadedServer.get_configured_server(log_app, host="localhost", port=13774)
with app_server.run_in_thread():
    with log_server.run_in_thread():
        tray = Tray()
        webbrowser.open("https://app.letta.com")
        tray.create()
