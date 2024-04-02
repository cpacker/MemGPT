import os

from sqlalchemy import create_engine, MetaData


def main():
    uri = os.environ.get(
        "MEMGPT_PGURI",
        "postgresql+pg8000://memgpt:memgpt@localhost:8888/memgpt",
    )

    engine = create_engine(uri)
    meta = MetaData()
    meta.reflect(bind=engine)
    meta.drop_all(bind=engine)


if __name__ == "__main__":
    main()
