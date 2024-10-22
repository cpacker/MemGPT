import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def extract_file_metadata(file_path) -> dict:
    """Extracts metadata from a single file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    file_metadata = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": mimetypes.guess_type(file_path)[0] or "unknown",
        "file_size": os.path.getsize(file_path),
        "file_creation_date": datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d"),
        "file_last_modified_date": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d"),
    }
    return file_metadata


def extract_metadata_from_files(file_list):
    """Extracts metadata for a list of files."""
    metadata = []
    for file_path in file_list:
        file_metadata = extract_file_metadata(file_path)
        if file_metadata:
            metadata.append(file_metadata)
    return metadata


def get_filenames_in_dir(
    input_dir: str, recursive: bool = True, required_exts: Optional[List[str]] = None, exclude: Optional[List[str]] = None
):
    """
    Recursively reads files from the directory, applying required_exts and exclude filters.
    Ensures that required_exts and exclude do not overlap.

    Args:
        input_dir (str): The directory to scan for files.
        recursive (bool): Whether to scan directories recursively.
        required_exts (list): List of file extensions to include (e.g., ['pdf', 'txt']).
                             If None or empty, matches any file extension.
        exclude (list): List of file patterns to exclude (e.g., ['*png', '*jpg']).

    Returns:
        list: A list of matching file paths.
    """
    required_exts = required_exts or []
    exclude = exclude or []

    # Ensure required_exts and exclude do not overlap
    ext_set = set(required_exts)
    exclude_set = set(exclude)
    overlap = ext_set & exclude_set
    if overlap:
        raise ValueError(f"Extensions in required_exts and exclude overlap: {overlap}")

    def is_excluded(file_name):
        """Check if a file matches any pattern in the exclude list."""
        for pattern in exclude:
            if Path(file_name).match(pattern):
                return True
        return False

    files = []
    search_pattern = "**/*" if recursive else "*"

    for file_path in Path(input_dir).glob(search_pattern):
        if file_path.is_file() and not is_excluded(file_path.name):
            ext = file_path.suffix.lstrip(".")
            # If required_exts is empty, match any file
            if not required_exts or ext in required_exts:
                files.append(file_path)

    return files


def assert_all_files_exist_locally(file_paths: List[str]) -> bool:
    """
    Checks if all file paths in the provided list exist locally.
    Raises a FileNotFoundError with a list of missing files if any do not exist.

    Args:
        file_paths (List[str]): List of file paths to check.

    Returns:
        bool: True if all files exist, raises FileNotFoundError if any file is missing.
    """
    missing_files = [file_path for file_path in file_paths if not Path(file_path).exists()]

    if missing_files:
        raise FileNotFoundError(missing_files)

    return True
