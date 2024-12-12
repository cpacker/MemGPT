from typing import Optional, Union

import base64
import numpy as np
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3

from letta.constants import MAX_EMBEDDING_DIM

def adapt_array(arr):
    """
    Converts numpy array to binary for SQLite storage
    """
    if arr is None:
        return None

    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.float32)
    elif not isinstance(arr, np.ndarray):
        raise ValueError(f"Unsupported type: {type(arr)}")
    
    # Convert to bytes and then base64 encode
    bytes_data = arr.tobytes()
    base64_data = base64.b64encode(bytes_data)
    return sqlite3.Binary(base64_data)

def convert_array(text):
    """
    Converts binary back to numpy array
    """
    if text is None:
        return None
    if isinstance(text, list):
        return np.array(text, dtype=np.float32)
    if isinstance(text, np.ndarray):
        return text

    # Handle both bytes and sqlite3.Binary
    binary_data = bytes(text) if isinstance(text, sqlite3.Binary) else text
    
    try:
        # First decode base64
        decoded_data = base64.b64decode(binary_data)
        # Then convert to numpy array
        return np.frombuffer(decoded_data, dtype=np.float32)
    except Exception as e:
        return None

def verify_embedding_dimension(embedding: np.ndarray, expected_dim: int = MAX_EMBEDDING_DIM) -> bool:
    """
    Verifies that an embedding has the expected dimension
    
    Args:
        embedding: Input embedding array
        expected_dim: Expected embedding dimension (default: 4096)
        
    Returns:
        bool: True if dimension matches, False otherwise
    """
    if embedding is None:
        return False
    return embedding.shape[0] == expected_dim

def validate_and_transform_embedding(
    embedding: Union[bytes, sqlite3.Binary, list, np.ndarray],
    expected_dim: int = MAX_EMBEDDING_DIM,
    dtype: np.dtype = np.float32
) -> Optional[np.ndarray]:
    """
    Validates and transforms embeddings to ensure correct dimensionality.
    
    Args:
        embedding: Input embedding in various possible formats
        expected_dim: Expected embedding dimension (default 4096)
        dtype: NumPy dtype for the embedding (default float32)
        
    Returns:
        np.ndarray: Validated and transformed embedding
        
    Raises:
        ValueError: If embedding dimension doesn't match expected dimension
    """
    if embedding is None:
        return None
        
    # Convert to numpy array based on input type
    if isinstance(embedding, (bytes, sqlite3.Binary)):
        vec = convert_array(embedding)
    elif isinstance(embedding, list):
        vec = np.array(embedding, dtype=dtype)
    elif isinstance(embedding, np.ndarray):
        vec = embedding.astype(dtype)
    else:
        raise ValueError(f"Unsupported embedding type: {type(embedding)}")
    
    # Validate dimension
    if vec.shape[0] != expected_dim:
        raise ValueError(
            f"Invalid embedding dimension: got {vec.shape[0]}, expected {expected_dim}"
        )
    
    return vec

def cosine_distance(embedding1, embedding2, expected_dim=MAX_EMBEDDING_DIM):
    """
    Calculate cosine distance between two embeddings
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        expected_dim: Expected embedding dimension (default 4096)
        
    Returns:
        float: Cosine distance
    """
    
    if embedding1 is None or embedding2 is None:
        return 0.0  # Maximum distance if either embedding is None
    
    try:
        vec1 = validate_and_transform_embedding(embedding1, expected_dim)
        vec2 = validate_and_transform_embedding(embedding2, expected_dim)
    except ValueError as e:
        return 0.0
        
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    distance = float(1.0 - similarity)
    
    return distance

@event.listens_for(Engine, "connect")
def register_functions(dbapi_connection, connection_record):
    """Register SQLite functions"""
    if isinstance(dbapi_connection, sqlite3.Connection):
        dbapi_connection.create_function("cosine_distance", 2, cosine_distance)
    
# Register adapters and converters for numpy arrays
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)
