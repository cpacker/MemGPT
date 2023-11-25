from sqlalchemy import create_engine, Column, Integer, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_json import mutable_json_type
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_json import mutable_json_type

from memgpt.connectors.storage import StorageConnector
from memgpt.config import MemGPTConfig

Base = declarative_base()

def get_persistence_model(table_name: str):
    class PersistenceModel(Base):
        __abstract__ = True  # this line is necessary
        id = Column(Integer, primary_key=True)
        message = Column(mutable_json_type(dbtype=JSONB, nested=True))

    """Create database model for table_name"""
    class_name = f"{table_name.capitalize()}Model"
    Model = type(class_name, (PersistenceModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
    return Model

class PostgresPersistenceConnector(StorageConnector):
    def __init__(self, name):
        self.name = name
        config = MemGPTConfig.load()
        engine = create_engine(config.archival_storage_uri)     #FIXME: create & use persistence_storage_uri
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


    # def save_message(self, message):
    #     q = self.persistence_session.query(PersistenceModel).filter(PersistenceModel.data['name'].astext == data['name'])
    #     doc = q.first()
    #     if doc is None:
    #         doc = PersistenceModel(data=data)
    #         self.persistence_session.add(doc)
    #     doc.data = data
    #     self.persistence_session.commit()

    # def test(self):
    #     sq.save_persistence_data({'foo': "bar", 'name': "baz"})
    #     result = sq.load_persistence_data("baz")
    #     print("RESULT (should be same as data):", result)
    #     agents = sq.get_agent_list()
    #     print ("AGENTS:", agents)

if __name__ == "__main__":
    con = PostgresPersistenceConnector('ag0')
    mod = get_persistence_model(con.name)
    doc = mod()
    doc.message={"msg": "MEssage to yuo TRUDY"}
    con.session.add(doc)
    con.session.commit()