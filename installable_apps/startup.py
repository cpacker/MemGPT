import pgserver
import webbrowser

from letta.settings import settings

from server import ThreadedServer
from tray import Tray

pgdata = settings.letta_dir / "pgdata"
pgdata.mkdir(parents=True, exist_ok=True)

database = pgserver.get_server(pgdata)
# create pg vector extension
#database.psql('CREATE EXTENSION IF NOT EXISTS pg_vector;')

# feed database URI parts to the application
settings.pg_uri = database.get_uri()
# start the server
app_server = ThreadedServer.get_configured_server()

with app_server.run_in_thread():
    tray = Tray()
    webbrowser.open("https://app.letta.com")
    tray.create()
