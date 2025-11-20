"""Temporary file management utilities for DBS Toolbox.

This module provides utilities for managing temporary files with session-based
organization and automatic cleanup.
"""

from pathlib import Path
from typing import Optional
import uuid
import tempfile


# Global temp directory reference (set by main.py)
_TEMP_DIR: Optional[Path] = None


def set_temp_dir(temp_dir: Path):
    """Set the global temporary directory.

    This should be called once by main.py during initialization.

    Args:
        temp_dir: Path to the temporary directory
    """
    global _TEMP_DIR
    _TEMP_DIR = temp_dir


def get_temp_dir() -> Path:
    """Get the global temporary directory.

    Returns:
        Path to the temporary directory

    Raises:
        RuntimeError: If temp directory hasn't been initialized
    """
    if _TEMP_DIR is None:
        raise RuntimeError("Temp directory not initialized. Call set_temp_dir() first.")
    return _TEMP_DIR


def get_session_dir(session_id: Optional[str] = None) -> Path:
    """Get or create a session-specific temporary directory.

    Args:
        session_id: Optional session ID. If not provided, generates a new UUID.

    Returns:
        Path to the session directory
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    session_dir = get_temp_dir() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_uploaded_file(file_content: bytes, file_name: str, session_id: Optional[str] = None) -> Path:
    """Save an uploaded file to a session directory.

    If a file with the same name already exists, adds a numeric suffix to make it unique.

    Args:
        file_content: Binary content of the file
        file_name: Name of the file
        session_id: Optional session ID. If not provided, uses a new session.

    Returns:
        Path to the saved file
    """
    session_dir = get_session_dir(session_id)
    file_path = session_dir / file_name

    # If file already exists, add a numeric suffix to make it unique
    if file_path.exists():
        # Handle double extensions like .nii.gz
        if file_name.endswith('.nii.gz'):
            stem = file_name[:-7]  # Remove .nii.gz
            suffix = '.nii.gz'
        else:
            stem = file_path.stem
            suffix = file_path.suffix

        # Find a unique name by adding a counter
        counter = 1
        while file_path.exists():
            new_name = f"{stem}_{counter}{suffix}"
            file_path = session_dir / new_name
            counter += 1

    file_path.write_bytes(file_content)
    return file_path


def get_session_file_path(file_name: str, session_id: Optional[str] = None) -> Path:
    """Get the path for a file in a session directory (without creating it).

    Args:
        file_name: Name of the file
        session_id: Optional session ID. If not provided, uses a new session.

    Returns:
        Path where the file should be saved
    """
    session_dir = get_session_dir(session_id)
    return session_dir / file_name


def cleanup_session(session_id: str):
    """Clean up a specific session directory.

    Args:
        session_id: Session ID to clean up
    """
    import shutil

    session_dir = get_temp_dir() / session_id
    if session_dir.exists() and session_dir.is_dir():
        try:
            shutil.rmtree(session_dir)
            print(f"Cleaned up session directory: {session_id}")
        except Exception as e:
            print(f"Error cleaning session directory {session_id}: {e}")
