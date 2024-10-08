import functools
import os
import time

from tests.helpers.endpoints_helper import (
    check_agent_archival_memory_retrieval,
    check_agent_edit_core_memory,
    check_agent_recall_chat_memory,
    check_agent_uses_external_tool,
    check_first_response_is_valid_for_llm_endpoint,
    check_response_contains_keyword,
    run_embedding_endpoint,
)

# directories
embedding_config_dir = "configs/embedding_model_configs"
llm_config_dir = "tests/configs/llm_model_configs"


def retry_until_threshold(threshold=0.5, max_attempts=10, sleep_time_seconds=4):
    """
    Decorator to retry a test until a failure threshold is crossed.

    :param threshold: Expected passing rate (e.g., 0.5 means 50% success rate expected).
    :param max_attempts: Maximum number of attempts to retry the test.
    """

    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            success_count = 0
            failure_count = 0

            for attempt in range(max_attempts):
                try:
                    func(*args, **kwargs)
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    print(f"\033[93mAn attempt failed with error:\n{e}\033[0m")

                time.sleep(sleep_time_seconds)

            rate = success_count / max_attempts
            if rate >= threshold:
                print(f"Test met expected passing rate of {threshold:.2f}. Actual rate: {success_count}/{max_attempts}")
            else:
                raise AssertionError(
                    f"Test did not meet expected passing rate of {threshold:.2f}. Actual rate: {success_count}/{max_attempts}"
                )

        return wrapper

    return decorator_retry


# ======================================================================================================================
# OPENAI TESTS
# ======================================================================================================================
def test_openai_gpt_4_returns_valid_first_message():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    response = check_first_response_is_valid_for_llm_endpoint(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_openai_gpt_4_returns_keyword():
    keyword = "banana"
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    response = check_response_contains_keyword(filename, keyword=keyword)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_openai_gpt_4_uses_external_tool():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    response = check_agent_uses_external_tool(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_openai_gpt_4_recall_chat_memory():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    response = check_agent_recall_chat_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_openai_gpt_4_archival_memory_retrieval():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    response = check_agent_archival_memory_retrieval(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_openai_gpt_4_edit_core_memory():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    response = check_agent_edit_core_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_embedding_endpoint_openai():
    filename = os.path.join(embedding_config_dir, "text-embedding-ada-002.json")
    run_embedding_endpoint(filename)


# ======================================================================================================================
# AZURE TESTS
# ======================================================================================================================
def test_azure_gpt_4o_mini_returns_valid_first_message():
    filename = os.path.join(llm_config_dir, "azure-gpt-4o-mini.json")
    response = check_first_response_is_valid_for_llm_endpoint(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_azure_gpt_4o_mini_returns_keyword():
    keyword = "banana"
    filename = os.path.join(llm_config_dir, "azure-gpt-4o-mini.json")
    response = check_response_contains_keyword(filename, keyword=keyword)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_azure_gpt_4o_mini_uses_external_tool():
    filename = os.path.join(llm_config_dir, "azure-gpt-4o-mini.json")
    response = check_agent_uses_external_tool(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_azure_gpt_4o_mini_recall_chat_memory():
    filename = os.path.join(llm_config_dir, "azure-gpt-4o-mini.json")
    response = check_agent_recall_chat_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_azure_gpt_4o_mini_archival_memory_retrieval():
    filename = os.path.join(llm_config_dir, "azure-gpt-4o-mini.json")
    response = check_agent_archival_memory_retrieval(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_azure_gpt_4o_mini_edit_core_memory():
    filename = os.path.join(llm_config_dir, "azure-gpt-4o-mini.json")
    response = check_agent_edit_core_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


# ======================================================================================================================
# LETTA HOSTED
# ======================================================================================================================
def test_llm_endpoint_letta_hosted():
    filename = os.path.join(llm_config_dir, "letta-hosted.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_letta_hosted():
    filename = os.path.join(embedding_config_dir, "letta-hosted.json")
    run_embedding_endpoint(filename)


# ======================================================================================================================
# LOCAL MODELS
# ======================================================================================================================
def test_embedding_endpoint_local():
    filename = os.path.join(embedding_config_dir, "local.json")
    run_embedding_endpoint(filename)


def test_llm_endpoint_ollama():
    filename = os.path.join(llm_config_dir, "ollama.json")
    check_first_response_is_valid_for_llm_endpoint(filename)


def test_embedding_endpoint_ollama():
    filename = os.path.join(embedding_config_dir, "ollama.json")
    run_embedding_endpoint(filename)


# ======================================================================================================================
# ANTHROPIC TESTS
# ======================================================================================================================
def test_claude_opus_3_returns_valid_first_message():
    filename = os.path.join(llm_config_dir, "claude-3-opus.json")
    response = check_first_response_is_valid_for_llm_endpoint(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_claude_opus_3_returns_keyword():
    keyword = "banana"
    filename = os.path.join(llm_config_dir, "claude-3-opus.json")
    response = check_response_contains_keyword(filename, keyword=keyword)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_claude_opus_3_uses_external_tool():
    filename = os.path.join(llm_config_dir, "claude-3-opus.json")
    response = check_agent_uses_external_tool(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_claude_opus_3_recall_chat_memory():
    filename = os.path.join(llm_config_dir, "claude-3-opus.json")
    response = check_agent_recall_chat_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_claude_opus_3_archival_memory_retrieval():
    filename = os.path.join(llm_config_dir, "claude-3-opus.json")
    response = check_agent_archival_memory_retrieval(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_claude_opus_3_edit_core_memory():
    filename = os.path.join(llm_config_dir, "claude-3-opus.json")
    response = check_agent_edit_core_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


# ======================================================================================================================
# GROQ TESTS
# ======================================================================================================================
def test_groq_llama31_70b_returns_valid_first_message():
    filename = os.path.join(llm_config_dir, "groq.json")
    response = check_first_response_is_valid_for_llm_endpoint(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_groq_llama31_70b_returns_keyword():
    keyword = "banana"
    filename = os.path.join(llm_config_dir, "groq.json")
    response = check_response_contains_keyword(filename, keyword=keyword)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_groq_llama31_70b_uses_external_tool():
    filename = os.path.join(llm_config_dir, "groq.json")
    response = check_agent_uses_external_tool(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_groq_llama31_70b_recall_chat_memory():
    filename = os.path.join(llm_config_dir, "groq.json")
    response = check_agent_recall_chat_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


@retry_until_threshold(threshold=0.75, max_attempts=4)
def test_groq_llama31_70b_archival_memory_retrieval():
    filename = os.path.join(llm_config_dir, "groq.json")
    response = check_agent_archival_memory_retrieval(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")


def test_groq_llama31_70b_edit_core_memory():
    filename = os.path.join(llm_config_dir, "groq.json")
    response = check_agent_edit_core_memory(filename)
    # Log out successful response
    print(f"Got successful response from client: \n\n{response}")
