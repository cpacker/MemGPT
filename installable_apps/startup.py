from sys import exit
import pgserver

from letta.settings import settings
from letta.server.rest_api.app import start_server

# TODO: pull from the app config stack
pgdata = settings.letta_dir / "pgdata"
pgdata.mkdir(parents=True, exist_ok=True)

database = pgserver.get_server(pgdata)
# create pg vector extension
database.psql('CREATE EXTENSION IF NOT EXISTS pg_vector;')

# feed database URI parts to the application
settings.pg_uri = database.get_uri()

# start the server
try:
    start_server()
except (KeyboardInterrupt, SystemExit):
    # TODO: how does the application close signal manifest? SystemExit? Confirm this.
    exit(0)
