from sqlalchemy import create_engine, MetaData

engine = create_engine("postgresql+pg8000://memgpt:memgpt@localhost:8888/memgpt")

meta = MetaData()
meta.reflect(bind=engine)

meta.drop_all(bind=engine)
