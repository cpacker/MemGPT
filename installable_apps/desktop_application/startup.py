#!/usr/bin/env python3
import pgserver
import webbrowser

from letta.settings import settings
from letta.server.rest_api.app import app as letta_app
from letta.server.constants import REST_DEFAULT_PORT

from desktop_application.server import ThreadedServer
from installable_apps.desktop_application.logserver import app as log_app
from desktop_application.tray import Tray
from desktop_application.installable_logger import get_logger

logger = get_logger(__name__)

def initialize_database():
    """Initialize the postgres binary database"""
    # create the pgdata
    logger.info("Initializing database...")
    pgdata = settings.letta_dir / "pgdata"
    pgdata.mkdir(parents=True, exist_ok=True)

    try:
        database = pgserver.get_server(pgdata)
        # create pg vector extension
        database.psql('CREATE EXTENSION IF NOT EXISTS vector')
        logger.info("Database initialized at %s", pgdata)
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        raise e
    logger.debug("Configuring app with databsase uri...")
    # feed database URI parts to the application
    settings.pg_uri = database.get_uri()
    logger.debug("Database URI: %s configured in settings", settings.pg_uri)

def run_servers():
    """launch letta and letta logs"""
    app_server = ThreadedServer.get_configured_server(letta_app, host="localhost", port=REST_DEFAULT_PORT)
    log_server = ThreadedServer.get_configured_server(log_app, host="localhost", port=13774)
    with app_server.run_in_thread():
        logger.info("App server started")
        with log_server.run_in_thread():
            logger.info("Log server started")
            tray = Tray()
            logger.info("Tray created")
            webbrowser.open("https://app.letta.com")
            tray.create()

## execute
initialize_database()
run_servers()