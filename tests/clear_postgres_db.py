import os

from sqlalchemy import MetaData, create_engine


def main():
    uri = os.environ.get(
        "MEMGPT_PGURI",
        "postgresql+pg8000://letta:letta@localhost:8888/letta",
    )

    engine = create_engine(uri)
    meta = MetaData()
    meta.reflect(bind=engine)
    meta.drop_all(bind=engine)


if __name__ == "__main__":
    main()
