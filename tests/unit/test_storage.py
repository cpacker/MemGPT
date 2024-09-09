import random

from faker import Faker
from pytest import fixture
from pytest import mark as m

from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.orm.__all__ import Base
from memgpt.schemas.enums import MessageRole, TableType
from memgpt.schemas.message import MessageCreate
from memgpt.schemas.passage import PassageCreate

faker = Faker()


@m.describe("When working with StorageConnectors")
class TestUnitStorage:

    @fixture(params=[t for t in TableType if t != TableType.DOCUMENTS])  # Document is not currently supported
    def storage(self, request, user_and_agent_seed):
        config = MemGPTConfig.load()
        user, agent = user_and_agent_seed
        user = user.to_pydantic()
        agent = agent.to_pydantic()
        return StorageConnector.get_storage_connector(request.param, config, user.id, agent.id)

    def _storage_data(self, type: TableType, user_id: str = None, agent_id: str = None, count: int = 3) -> list:
        match type:
            case TableType.ARCHIVAL_MEMORY | TableType.PASSAGES:
                data = [
                    PassageCreate(
                        user_id=user_id,
                        agent_id=agent_id,
                        text=faker.text(),
                    )
                    for _ in range(count)
                ]
            case TableType.RECALL_MEMORY:
                data = [
                    MessageCreate(
                        agent_id=agent_id,
                        user_id=user_id,
                        role=random.choice(list(MessageRole)),
                        text=faker.text(),
                        name=faker.name(),
                    )
                    for _ in range(count)
                ]
            case _:
                return []
        return data

    @m.context("and choosing a table type")
    @m.it("should select an ORM model")
    def test_create_storage(self, storage):
        assert storage.SQLModel is not None
        assert type(storage.SQLModel) == type(Base)

    @m.context("and inserting many records")
    @m.it("should add all records")
    def test_insert_storage(self, storage, db_session):
        _data = self._storage_data(storage.table_type, storage.user_id, storage.agent_id)

        storage.insert_many(records=_data)
        storage_records = storage.get_all()
        item_ids = [item.id for item in storage_records]

        with db_session as session:
            records = storage.SQLModel.list(db_session=session)

        assert len(records) == len(_data) == len(storage_records)

        random_item = storage.get(random.choice(item_ids))
        assert random_item is not None
        assert random_item.text in [item.text for item in _data]
