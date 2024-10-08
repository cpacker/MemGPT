import os

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
def test_llm_endpoint_groq():
    filename = os.path.join(llm_config_dir, "groq.json")
    check_first_response_is_valid_for_llm_endpoint(filename)
