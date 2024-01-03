import os
import sys
import pickle
import argparse
from pprint import pp
from sqlalchemy import create_engine, Column, Integer, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_json import mutable_json_type

from memgpt.connectors.storage import StorageConnector
from memgpt.config import MemGPTConfig

Base = declarative_base()


class PersistenceModel(Base):
    __abstract__ = True  # this line is necessary
    id = Column(Integer, primary_key=True)
    message = Column(mutable_json_type(dbtype=JSONB, nested=True))


def get_persistence_model(table_name: str):
    """Create database model for table_name. Put model in global space to avoid SqlAlchemy warnings"""
    class_name = f"{table_name.capitalize()}Model"
    if table_name not in list(Base.metadata.tables.keys()):
        Model = type(
            class_name, (PersistenceModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}}
        )
        globals()[class_name] = Model
    return globals()[class_name]


class PostgresPersistenceConnector(StorageConnector):
    def __init__(self, name):
        self.name = name
        config = MemGPTConfig.load()
        engine = create_engine(config.archival_storage_uri)  # FIXME: create & use persistence_storage_uri
        Base.metadata.create_all(engine)  # Create the table if it doesn't exist
        self.session = sessionmaker(bind=engine)()
        self.class_name = f"{name.capitalize()}Model"

        # create table
        self.uri = config.archival_storage_uri
        if config.archival_storage_uri is None:
            raise ValueError(f"Must specifiy archival_storage_uri in config {config.config_path}")
        self.db_model = get_persistence_model(self.name)
        self.engine = create_engine(self.uri)
        Base.metadata.create_all(self.engine)  # Create the table if it doesn't exist
        self.Session = sessionmaker(bind=self.engine)
        self.Session().execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Enables the vector extension


def main(name):
    persistence_dir = f"{os.path.expanduser('~/')}.memgpt/agents/{name}/persistence_manager/"
    print(persistence_dir)
    con = PostgresPersistenceConnector(name)
    mod = con.db_model
    try:
        persistence_files = os.listdir(persistence_dir)
        for pfile in persistence_files:
            print("persistence_manager pickle file:", pfile)
            with open(persistence_dir + pfile, "rb") as fh:
                pers = pickle.load(fh)
                # print (pers.keys())
                for msg in pers["all_messages"]:
                    # print ("Processing message:", msg['timestamp'])
                    q = con.session.query(mod).filter(mod.message == msg)
                    if q.first() is not None:
                        print(f"Identical message with timestamp {msg['timestamp']} already exists, not saving")
                    else:
                        doc = mod()
                        doc.message = msg
                        con.session.add(doc)
                        print(f"New message with timestamp {msg['timestamp']} saved")
                con.session.commit()
    except FileNotFoundError:
        print("No persistence_manager found for", name)


def search_role(name, role):
    con = PostgresPersistenceConnector(name)
    mod = con.db_model
    q = con.session.query(mod).filter(mod.message["message"]["role"].astext == role)
    ret = [doc.message for doc in q]
    # pp(ret)
    return ret


if __name__ == "__main__":
    # main(sys.argv[1])
    parser = argparse.ArgumentParser(description="export persistence data to postgres")
    parser.add_argument("--agent", required=True, help="Name of agent whose data is exported")
    parser.add_argument("--role", help="Return all messages with specified role")
    args = parser.parse_args()
    if args.role:
        pp(search_role(args.agent, args.role))
    else:
        main(args.agent)
