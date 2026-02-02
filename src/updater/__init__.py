"""
HelixOne Auto-Updater Module
Handles automatic update checking, downloading, and installation
"""

from .auto_updater import AutoUpdater, check_for_updates_async
from .version import CURRENT_VERSION, get_version_info

__all__ = [
    'AutoUpdater',
    'check_for_updates_async',
    'CURRENT_VERSION',
    'get_version_info',
]
