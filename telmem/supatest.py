import asyncio
from db import get_user_api_key, save_user_api_key, save_user_agent_id, get_user_agent_id

async def test_db_functions():
    # Define test data
    test_telegram_user_id = 123456789
    test_api_key = "test_api_key_123"
    test_agent_id = "test_agent_id_123"

    # Insert test data using custom db functions
    await save_user_api_key(test_telegram_user_id, test_api_key)
    await save_user_agent_id(test_telegram_user_id, test_agent_id)

    # Fetch inserted data using custom db functions
    fetched_api_key = await get_user_api_key(test_telegram_user_id)
    fetched_agent_id = await get_user_agent_id(test_telegram_user_id)

    # Validate insert and fetch operations
    assert fetched_api_key == test_api_key, "API Key mismatch"
    assert fetched_agent_id == test_agent_id, "Agent ID mismatch"

    print("Custom DB function tests passed successfully.")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_db_functions())
