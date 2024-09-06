from pytest import mark as m


@m.describe("When interacting with MemGPT via the command line tool")
class TestCLIClient:

    @m.context("and this is a user's first time using the tool")
    @m.it("should walk the user through setup of MemGPT")
    async def test_cli_initial_setup(self):
        assert False

    @m.context("and the user has already set up MemGPT")
    @m.it("should direct the user to select an existing agent or create a new one")
    async def test_cli_select_or_create_agent(self):
        assert False

    @m.context("and the user has selected an existing agent")
    @m.it("should allow the user to interact with the agent")
    async def test_cli_interact_with_existing_agent(self):
        assert False

    @m.context("and the user has created a new agent")
    @m.it("should allow the user to interact with the agent")
    async def test_cli_interact_with_new_agent(self):
        assert False

    @m.context("and the user has selected an existing agent")
    @m.it("should allow the user to rename the agent")
    async def test_cli_rename_agent(self):
        assert False

    @m.context("and the user has selected an existing agent")
    @m.it("should allow the user to delete the agent")
    async def test_cli_delete_agent(self):
        assert False

    @m.context("and the user has created a new agent")
    @m.it("should allow the user to persist (save) the agent")
    async def test_cli_persist_agent(self):
        assert False
