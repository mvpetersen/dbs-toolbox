"""File upload component."""

from nicegui import ui
from typing import Callable, Optional


def create_file_upload(
    on_upload: Callable,
    accept: str = '.nii,.nii.gz',
    label: str = 'Upload file',
    max_file_size: int = 500 * 1024 * 1024  # 500MB default
) -> ui.upload:
    """Create a file upload component.
    
    Args:
        on_upload: Callback function when file is uploaded
        accept: File types to accept
        label: Label for the upload area
        max_file_size: Maximum file size in bytes
        
    Returns:
        ui.upload component
    """
    return ui.upload(
        on_upload=on_upload,
        max_file_size=max_file_size,
        label=label,
        auto_upload=True
    ).classes('w-full').props(f'accept="{accept}" square outlined')