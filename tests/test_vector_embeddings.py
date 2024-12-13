import numpy as np

from letta.orm.sqlalchemy_base import adapt_array
from letta.orm.sqlite_functions import convert_array, verify_embedding_dimension


def test_vector_conversions():
    """Test the vector conversion functions"""
    # Create test data
    original = np.random.random(4096).astype(np.float32)
    print(f"Original shape: {original.shape}")

    # Test full conversion cycle
    encoded = adapt_array(original)
    print(f"Encoded type: {type(encoded)}")
    print(f"Encoded length: {len(encoded)}")

    decoded = convert_array(encoded)
    print(f"Decoded shape: {decoded.shape}")
    print(f"Dimension verification: {verify_embedding_dimension(decoded)}")

    # Verify data integrity
    np.testing.assert_array_almost_equal(original, decoded)
    print("✓ Data integrity verified")

    # Test with a list
    list_data = original.tolist()
    encoded_list = adapt_array(list_data)
    decoded_list = convert_array(encoded_list)
    np.testing.assert_array_almost_equal(original, decoded_list)
    print("✓ List conversion verified")

    # Test None handling
    assert adapt_array(None) is None
    assert convert_array(None) is None
    print("✓ None handling verified")


# Run the tests
