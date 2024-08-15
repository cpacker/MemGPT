"""
Tests for the Python client

The following basic functionality should be tested within the smoke tests:

Agents
- create_agent
- update_agent
  - system prompt
  - core memory
  - message IDs
  - tools
  - name
  - description
- delete_agent
- get_agent_id
- get_agent
- list_agents

Sources
- create_source
- update_source
- delete_source
- attach_source
- get_source_id
- get_source
- list_sources

Memory
- get_memory
- get_archival_memory_summary
- get_recall_memory_summary
- get_in_context_messages
- get_messages_paginated
- get_passages_paginated

Memory Blocks
- create_block (replaces human/persona)
- delete_block (replaces human/persona)
- update_block (replaces human/persona)
- get_block_id
- get_block
- list_blocks

Tools
- create_tool
- update_tool
- delete_tool
- get_tool_id
- get_tool
- list_tools


The following advanced functionality should be tested within the smoke tests:

- concurrent requests
    - single agent (multiple users)
    - multiple agents (multiple users)

- memory sharing
    - multiple agents sharing a memory block

- custom memory types
    - extending/replacing the default `ChatMemory` class (tools, memory schema)

- streaming

- tool integration tests

"""

from pytest import mark as m


@m.describe("Smoke tests for the Python client")
class TestPythonClient:

    # Agents

    @m.context("a client has been created and successfully connected to the server")
    @m.it("should create an agent")
    def test_create_agent(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should update an agent's system prompt")
    def test_update_agent_system_prompt(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should update an agent's core memory")
    def test_update_agent_core_memory(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should update an agent's message IDs")
    def test_update_agent_message_ids(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should update an agent's tools")
    def test_update_agent_tools(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should update an agent's name")
    def test_update_agent_name(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should update an agent's description")
    def test_update_agent_description(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should delete an agent")
    def test_delete_agent(self):

        # should also delete data in the recall/archival memory store
        # should delete blocks attached to the agent
        pass

    @m.context("a agent has already been created")
    @m.it("should get an agent ID")
    def test_get_agent_id(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should get an agent")
    def test_get_agent(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should list agents")
    def test_list_agents(self):
        pass

    # Sources

    @m.context("a client has been created and successfully connected to the server")
    @m.it("should create a source")
    def test_create_source(self):
        pass

    @m.context("a source has already been created")
    @m.it("should update a source")
    def test_update_source_name(self):
        pass

    @m.context("a source has already been created")
    @m.it("should add passages to a source")
    def test_add_passages(self):
        pass

    @m.context("a source has already been created")
    @m.it("should delete a source")
    def test_delete_source(self):
        pass

    @m.it("should list sources")
    def test_list_sources(self):
        # TODO: test global sources?
        # TODO: list sources available to user
        pass

    # Data connectors

    @m.it("should create a data connector")
    def test_create_data_connector(self):
        pass

    @m.context("a source and data connector has already been created")
    @m.it("should load a data connector into a source")
    def test_load_data_connector(self):
        pass

    # memory

    @m.context("a agent has already been created")
    @m.it("should get the agent's memory")
    def test_get_memory(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should get the agent's archival memory summary")
    def test_get_archival_memory_summary(self):
        pass

    @m.context("a agent has already been created")
    @m.it("should get the agent's recall memory summary")
    def test_get_recall_memory_summary(self):

        # TODO: check that the summary is accurate
        pass

    @m.context("a agent has already been created and been messaged multiple times")
    @m.it("should get the agent's in-context messages")
    def test_get_in_context_messages(self):
        # TODO: should test to make sure the message IDs are accurate
        # the in-context messages are the most recent messages that are contained in the LLM context window
        pass

    @m.context()
    @m.it("should get messages paginated")
    def test_get_messages_paginated(self):
        # TODO: should check accurate pagination
        pass

    @m.context("a agent has already been created and has archival messages and/or attached sources")
    @m.it("should get passages paginated")
    def test_get_passages_paginated(self):
        # TODO: should check accurate pagination
        # passages are either in 1. an agent's archival memory 2. in a data source - probably both should be tested
        pass

    # Blocks

    @m.it("should create a block")
    def test_create_block(self):
        pass

    @m.it("should delete a block")
    def test_delete_block(self):
        pass

    @m.it("should update a block's character limit")
    def test_update_block_limit(self):
        pass

    @m.it("should update a block's value")
    def test_update_block_value(self):
        pass

    @m.it("should get a block ID")
    def test_get_block_id(self):
        pass

    @m.it("should get a block")
    def test_get_block(self):
        pass

    @m.it("should list blocks")
    def test_list_blocks(self):
        pass
