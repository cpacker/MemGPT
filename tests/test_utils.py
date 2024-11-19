import unicodedata

import pytest

from letta.constants import MAX_FILENAME_LENGTH, RESERVED_FILENAMES
from letta.utils import sanitize_filename


def test_valid_filename():
    filename = "valid_filename.txt"
    sanitized = sanitize_filename(filename)
    assert sanitized.startswith("valid_filename_")
    assert sanitized.endswith(".txt")


def test_filename_with_special_characters():
    filename = "invalid:/<>?*ƒfilename.txt"
    sanitized = sanitize_filename(filename)
    assert sanitized.startswith("_ƒfilename_")
    assert sanitized.endswith(".txt")


def test_filename_with_unicode():
    filename = "filē_with_ünîcødé.txt"
    sanitized = sanitize_filename(filename)
    normalized_filename = unicodedata.normalize("NFKD", filename).split(".")[0]
    assert sanitized.startswith(normalized_filename)
    assert sanitized.endswith(".txt")


def test_null_byte_in_filename():
    filename = "valid\0filename.txt"
    sanitized = sanitize_filename(filename)
    assert "\0" not in sanitized
    assert sanitized.startswith("valid_filename_")
    assert sanitized.endswith(".txt")


def test_path_traversal_characters():
    filename = "../../etc/passwd"
    sanitized = sanitize_filename(filename)
    assert sanitized.startswith("passwd_")
    assert len(sanitized) <= MAX_FILENAME_LENGTH


def test_reserved_filenames():
    for reserved in RESERVED_FILENAMES:
        filename = f"{reserved}.txt"
        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename(filename)


def test_empty_filename():
    with pytest.raises(ValueError, match="Invalid filename"):
        sanitize_filename("")


def test_dot_as_filename():
    with pytest.raises(ValueError, match="Invalid filename"):
        sanitize_filename(".")


def test_dotdot_as_filename():
    with pytest.raises(ValueError, match="Invalid filename"):
        sanitize_filename("..")


def test_long_filename():
    filename = "a" * (MAX_FILENAME_LENGTH + 10) + ".txt"
    sanitized = sanitize_filename(filename)
    assert len(sanitized) <= MAX_FILENAME_LENGTH
    assert sanitized.endswith(".txt")


def test_unique_filenames():
    filename = "duplicate.txt"
    sanitized1 = sanitize_filename(filename)
    sanitized2 = sanitize_filename(filename)
    assert sanitized1 != sanitized2
    assert sanitized1.startswith("duplicate_")
    assert sanitized2.startswith("duplicate_")
    assert sanitized1.endswith(".txt")
    assert sanitized2.endswith(".txt")
