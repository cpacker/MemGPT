import asyncio
from db import check_user_exists, get_user_api_key, get_user_agent_id
from memgpt import create_memgpt_user, send_message_to_memgpt

async def test_db_and_memgpt_integration():
    # Define test data
    test_telegram_user_id = 123456729
    test_message_text = "Hello, MemGPT!"  # Define a test message

    # Ensure user does not exist before test
    user_exists_before = await check_user_exists(test_telegram_user_id)
    assert not user_exists_before, "User should not exist yet"

    # Create a new MemGPT user and agent, and save their details
    creation_response = await create_memgpt_user(test_telegram_user_id)
    assert "Your MemGPT agent has been created." == creation_response, f"Unexpected creation response: {creation_response}"

    # Fetch API key and agent ID from DB to ensure they were saved
    fetched_api_key = await get_user_api_key(test_telegram_user_id)
    fetched_agent_id = await get_user_agent_id(test_telegram_user_id)
    assert fetched_api_key is not None, "API Key was not saved"
    assert fetched_agent_id is not None, "Agent ID was not saved"

    # Send a message to MemGPT and expect a response
    memgpt_response = await send_message_to_memgpt(test_telegram_user_id, test_message_text)
    assert memgpt_response is not None and memgpt_response != "No API key or agent found. Please start again.", "Failed to get a response from MemGPT or no API key/agent found"

    # Check if user exists after all operations
    user_exists_after = await check_user_exists(test_telegram_user_id)
    assert user_exists_after, "User should exist after MemGPT user and agent creation"

    print("All DB and MemGPT integration tests passed successfully.")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_db_and_memgpt_integration())
