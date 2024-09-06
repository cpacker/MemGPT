from pytest import mark as m


@m.describe("When calling the MemGPT REST API")
class TestAPI:

    @m.context("and this is the first time the API has been initialized")
    @m.it("should provide the user with an API key via the command line")
    async def test_api_key_provided_to_user(self):
        assert False

    @m.context("and this is the first time the API has been initialized")
    @m.it("should already have a default user and organization scoped to the initial API key")
    async def test_default_user_org(self):
        assert False
