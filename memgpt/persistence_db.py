from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_json import mutable_json_type
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_json import mutable_json_type

import memgpt.config as cfg

Base = declarative_base()

def get_persistence_model(table_name: str):
    config = cfg.MemGPTConfig.load()

    class PersistenceModel(Base):
        __tablename__ = 'memgpt_persistence'
        id = Column(Integer, primary_key=True)
        messages = Column(mutable_json_type(dbtype=JSONB, nested=True))

    """Create database model for table_name"""
    class_name = f"{table_name.capitalize()}Model"
    Model = type(class_name, (PersistenceModel,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
    return Model

class Sqlalch:
    """
    sqlalchemy interface
    """
    persistence_session = None

    @classmethod
    def init(cls, name):
        if cls.persistence_session is None:
            config = cfg.MemGPTConfig.load()
            persistence_engine = create_engine(config.persistence_storage_uri)
            Base.metadata.create_all(persistence_engine)  # Create the table if it doesn't exist
            cls.persistence_session = sessionmaker(bind=persistence_engine)()
            cls.class_name = f"{name.capitalize()}Model"

    @classmethod
    def load_persistence_data(cls, name):
        cls.init(name)
        q = cls.persistence_session.query(PersistenceModel).filter(PersistenceModel.data['name'].astext == name)
        doc = q.first()
        return doc.data

    @classmethod
    def save_persistence_data(cls, data): #data must have a key: 'name'
        cls.init()
        q = cls.persistence_session.query(PersistenceModel).filter(PersistenceModel.data['name'].astext == data['name'])
        doc = q.first()
        if doc is None:
            doc = PersistenceModel(data=data)
            cls.persistence_session.add(doc)
        doc.data = data
        cls.persistence_session.commit()

    @classmethod
    def test(cls):
        sq.save_agent_data({'foo': "bar", 'name': "baz"})
        result = sq.load_agent_data("baz")
        print("RESULT (should be same as data):", result)
        agents = sq.get_agent_list()
        print ("AGENTS:", agents)

if __name__ == "__main__":
    sq = Sqlalch()
    sq.test()
