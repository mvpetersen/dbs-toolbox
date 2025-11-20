"""Notification utilities."""

from nicegui import ui
from typing import Optional


def notify_success(message: str, timeout: float = 3.0, position: str = 'top'):
    """Show success notification.
    
    Args:
        message: Notification message
        timeout: Time to show notification in seconds
        position: Position of notification
    """
    ui.notify(
        message,
        type='positive',
        position=position,
        timeout=timeout * 1000,
        close_button=True
    )


def notify_error(message: str, timeout: float = 5.0, position: str = 'top'):
    """Show error notification.
    
    Args:
        message: Error message
        timeout: Time to show notification in seconds
        position: Position of notification
    """
    ui.notify(
        message,
        type='negative',
        position=position,
        timeout=timeout * 1000,
        close_button=True
    )


def notify_info(message: str, timeout: float = 3.0, position: str = 'top'):
    """Show info notification.
    
    Args:
        message: Info message
        timeout: Time to show notification in seconds
        position: Position of notification
    """
    ui.notify(
        message,
        type='info',
        position=position,
        timeout=timeout * 1000,
        close_button=True
    )


def notify_warning(message: str, timeout: float = 4.0, position: str = 'top'):
    """Show warning notification.
    
    Args:
        message: Warning message
        timeout: Time to show notification in seconds
        position: Position of notification
    """
    ui.notify(
        message,
        type='warning',
        position=position,
        timeout=timeout * 1000,
        close_button=True
    )