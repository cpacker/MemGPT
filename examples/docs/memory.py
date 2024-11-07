from letta import BasicBlockMemory, create_client

client = create_client()


human_block = client.create_block(label="human", value="Name: Sarah")

sad_persona_block = client.create_block(label="persona", value="You are a sad and negative assistant :(")
sad_block_memory = BasicBlockMemory(human=human_block, persona=sad_persona_block)


happy_persona_block = client.create_block(label="persona", value="You are a happy and positive assistant!")
happy_block_memory = BasicBlockMemory(human=human_block, persona=happy_persona_block)

happy_agent = client.create_agent(name="happy_agent", memory=happy_block_memory)
sad_agent = client.create_agent(name="sad_agent", memory=sad_block_memory)

# Update the shared memory block (human) for happy agent
response = client.send_message(agent_id=happy_agent.id, role="user", message="my name is actually Charles")

# Use updated information
response = client.send_message(agent_id=sad_agent.id, role="user", message="whats my name?")


# creating block templates
