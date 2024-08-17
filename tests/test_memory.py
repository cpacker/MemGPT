import pytest

# Import the classes here, assuming the above definitions are in a module named memory_module
from memgpt.memory import BaseMemory, ChatMemory


@pytest.fixture
def sample_memory():
    return ChatMemory(persona="Chat Agent", human="User")


def test_create_chat_memory():
    """Test creating an instance of ChatMemory"""
    chat_memory = ChatMemory(persona="Chat Agent", human="User")
    assert chat_memory.memory["persona"].value == "Chat Agent"
    assert chat_memory.memory["human"].value == "User"


def test_dump_memory_as_json(sample_memory):
    """Test dumping ChatMemory as JSON compatible dictionary"""
    memory_dict = sample_memory.to_dict()
    assert isinstance(memory_dict, dict)
    assert "persona" in memory_dict
    assert memory_dict["persona"]["value"] == "Chat Agent"


def test_load_memory_from_json(sample_memory):
    """Test loading ChatMemory from a JSON compatible dictionary"""
    memory_dict = sample_memory.to_dict()
    print(memory_dict)
    new_memory = BaseMemory.load(memory_dict)
    assert new_memory.memory["persona"].value == "Chat Agent"
    assert new_memory.memory["human"].value == "User"


# def test_memory_functionality(sample_memory):
#    """Test memory modification functions"""
#    # Get memory functions
#    functions = get_memory_functions(ChatMemory)
#    # Test core_memory_append function
#    append_func = functions['core_memory_append']
#    print("FUNCTIONS", functions)
#    env = {}
#    env.update(globals())
#    for tool in functions:
#        # WARNING: name may not be consistent?
#        exec(tool.source_code, env)
#
#    print(exec)
#
#    append_func(sample_memory, 'persona', " is a test.")
#    assert sample_memory.memory['persona'].value == "Chat Agent\n is a test."
#    # Test core_memory_replace function
#    replace_func = functions['core_memory_replace']
#    replace_func(sample_memory, 'persona', " is a test.", " was a test.")
#    assert sample_memory.memory['persona'].value == "Chat Agent\n was a test."


def test_memory_limit_validation(sample_memory):
    """Test exceeding memory limit"""
    with pytest.raises(ValueError):
        ChatMemory(persona="x" * 3000, human="y" * 3000)

    with pytest.raises(ValueError):
        sample_memory.memory["persona"].value = "x" * 3000
