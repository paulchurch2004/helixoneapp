"""
HelixOne Auto-Updater
Checks for updates, downloads, and installs new versions automatically.
Supports macOS (.dmg) and Windows (.exe).
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
import threading
from typing import Optional, Callable, Dict, Any
from pathlib import Path

import requests

from .version import CURRENT_VERSION, is_update_available

logger = logging.getLogger(__name__)


class AutoUpdater:
    """
    Automatic update manager for HelixOne

    Usage:
        updater = AutoUpdater()
        if updater.check_for_updates():
            updater.show_update_dialog()
    """

    # Update manifest URL - JSON file with latest version info
    # Format: {"version": "1.1.0", "mandatory": false, "changelog": [...],
    #          "download_url_mac": "...", "download_url_windows": "..."}
    VERSION_URL = "https://helixone.fr/api/version.json"

    REQUEST_TIMEOUT = 10
    CHUNK_SIZE = 8192

    def __init__(self):
        self.current_version = CURRENT_VERSION
        self.remote_version_info: Optional[Dict] = None
        self._download_progress = 0
        self._download_cancelled = False

    def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """
        Check if a new version is available.

        Returns:
            Version info dict if update available, None otherwise
        """
        try:
            logger.info(f"Checking for updates... (current: {self.current_version})")

            response = requests.get(
                self.VERSION_URL,
                timeout=self.REQUEST_TIMEOUT,
                headers={'User-Agent': f'HelixOne/{self.current_version}'}
            )
            response.raise_for_status()

            self.remote_version_info = response.json()
            remote_version = self.remote_version_info.get('version', '0.0.0')

            if is_update_available(remote_version):
                logger.info(f"Update available: {remote_version}")
                return self.remote_version_info
            else:
                logger.info("No update available")
                return None

        except requests.exceptions.Timeout:
            logger.warning("Update check timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking for updates: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid version info received: {e}")
            return None

    def get_download_url(self) -> Optional[str]:
        """Get the download URL for the current platform."""
        if not self.remote_version_info:
            return None

        if sys.platform == 'darwin':
            return self.remote_version_info.get('download_url_mac')
        elif sys.platform == 'win32':
            return self.remote_version_info.get('download_url_windows')

        # Fallback
        return self.remote_version_info.get('download_url')

    def _get_installer_filename(self) -> str:
        """Get the appropriate installer filename for the current platform."""
        if sys.platform == 'darwin':
            return "HelixOne.dmg"
        else:
            return "HelixOne_Setup.exe"

    def download_update(
        self,
        download_url: str,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Optional[str]:
        """
        Download the update installer.

        Args:
            download_url: URL to download the installer from
            progress_callback: Optional callback(percent) for progress updates

        Returns:
            Path to downloaded installer, or None on failure
        """
        self._download_cancelled = False
        self._download_progress = 0

        try:
            logger.info(f"Downloading update from: {download_url}")

            temp_dir = tempfile.mkdtemp(prefix="helixone_update_")
            installer_path = os.path.join(temp_dir, self._get_installer_filename())

            response = requests.get(
                download_url,
                stream=True,
                timeout=300,
                headers={'User-Agent': f'HelixOne/{self.current_version}'}
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(installer_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    if self._download_cancelled:
                        logger.info("Download cancelled by user")
                        return None

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            self._download_progress = int((downloaded / total_size) * 100)
                            if progress_callback:
                                progress_callback(self._download_progress)

            logger.info(f"Download complete: {installer_path}")
            return installer_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading update: {e}")
            return None
        except IOError as e:
            logger.error(f"Error saving update file: {e}")
            return None

    def cancel_download(self):
        """Cancel an ongoing download"""
        self._download_cancelled = True

    def install_update(self, installer_path: str, silent: bool = True) -> bool:
        """
        Launch the installer and exit the current application.

        Args:
            installer_path: Path to the downloaded installer
            silent: Run installer in silent mode

        Returns:
            True if installer launched successfully
        """
        try:
            if not os.path.exists(installer_path):
                logger.error(f"Installer not found: {installer_path}")
                return False

            logger.info(f"Launching installer: {installer_path}")

            if sys.platform == 'darwin':
                # macOS: open the .dmg file (mounts it and shows in Finder)
                subprocess.Popen(['open', installer_path], close_fds=True)
                logger.info("DMG opened - user will drag to Applications")
                sys.exit(0)

            elif sys.platform == 'win32':
                cmd = [installer_path]
                if silent:
                    cmd.append('/SILENT')
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    close_fds=True
                )
                logger.info("Exiting for update installation...")
                sys.exit(0)

            else:
                # Linux or other
                subprocess.Popen([installer_path], close_fds=True)
                sys.exit(0)

        except Exception as e:
            logger.error(f"Error launching installer: {e}")
            return False

    def get_changelog(self) -> list:
        """Get changelog from remote version info"""
        if self.remote_version_info:
            return self.remote_version_info.get('changelog', [])
        return []

    def is_mandatory_update(self) -> bool:
        """Check if update is mandatory"""
        if self.remote_version_info:
            return self.remote_version_info.get('mandatory', False)
        return False


def check_for_updates_async(callback: Callable[[Optional[Dict]], None]):
    """
    Check for updates in background thread.

    Args:
        callback: Function to call with update info (or None if no update)
    """
    def _check():
        updater = AutoUpdater()
        result = updater.check_for_updates()
        callback(result)

    thread = threading.Thread(target=_check, daemon=True)
    thread.start()


def show_update_dialog_ctk(parent, version_info: Dict) -> bool:
    """
    Show modern update dialog using CustomTkinter.

    Args:
        parent: Parent CTk window
        version_info: Version info dict from check_for_updates()

    Returns:
        True if user chose to update
    """
    try:
        import customtkinter as ctk
        import tkinter as tk

        version = version_info.get('version', 'Unknown')
        changelog = version_info.get('changelog', [])
        mandatory = version_info.get('mandatory', False)

        result = {'update': False}

        # Créer une fenêtre toplevel moderne
        dialog = ctk.CTkToplevel(parent)
        dialog.title("Mise à jour disponible")
        dialog.geometry("500x400")
        dialog.resizable(False, False)

        # Centrer la fenêtre
        dialog.transient(parent)
        dialog.grab_set()

        # Forcer au premier plan
        dialog.lift()
        dialog.focus_force()

        # Frame principale
        main_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        title_label = ctk.CTkLabel(
            main_frame,
            text=f"🚀 Nouvelle version disponible : v{version}",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Changelog
        changelog_label = ctk.CTkLabel(
            main_frame,
            text="Nouveautés :",
            font=("Arial", 14, "bold"),
            anchor="w"
        )
        changelog_label.pack(anchor="w", pady=(0, 10))

        # Frame scrollable pour le changelog
        changelog_frame = ctk.CTkScrollableFrame(main_frame, height=150)
        changelog_frame.pack(fill="both", expand=True, pady=(0, 20))

        for item in changelog:
            item_label = ctk.CTkLabel(
                changelog_frame,
                text=f"• {item}",
                font=("Arial", 12),
                anchor="w",
                wraplength=400
            )
            item_label.pack(anchor="w", pady=2)

        # Message obligatoire si nécessaire
        if mandatory:
            mandatory_label = ctk.CTkLabel(
                main_frame,
                text="⚠️ Cette mise à jour est obligatoire",
                font=("Arial", 12, "bold"),
                text_color="#FF6B6B"
            )
            mandatory_label.pack(pady=(0, 10))

        # Boutons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))

        def on_update():
            result['update'] = True
            dialog.destroy()

        def on_cancel():
            if not mandatory:
                result['update'] = False
                dialog.destroy()

        update_btn = ctk.CTkButton(
            button_frame,
            text="Mettre à jour maintenant",
            command=on_update,
            fg_color="#4CAF50",
            hover_color="#45a049",
            height=40,
            font=("Arial", 14, "bold")
        )
        update_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        if not mandatory:
            cancel_btn = ctk.CTkButton(
                button_frame,
                text="Plus tard",
                command=on_cancel,
                fg_color="gray40",
                hover_color="gray30",
                height=40,
                font=("Arial", 14)
            )
            cancel_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Attendre la fermeture du dialogue
        dialog.wait_window()

        return result['update']

    except Exception as e:
        logger.error(f"Error showing update dialog: {e}")
        # Fallback sur messagebox basique
        try:
            from tkinter import messagebox
            version = version_info.get('version', 'Unknown')
            return messagebox.askyesno(
                "Mise à jour disponible",
                f"Version {version} disponible.\nMettre à jour maintenant?",
                parent=parent
            )
        except:
            return False


def show_download_progress_dialog(parent, updater: AutoUpdater, download_url: str):
    """
    Show download progress dialog with CustomTkinter.

    Args:
        parent: Parent CTk window
        updater: AutoUpdater instance
        download_url: URL to download from

    Returns:
        Path to downloaded installer, or None if cancelled/failed
    """
    try:
        import customtkinter as ctk
        import threading

        result = {'installer_path': None}

        # Créer le dialogue de progression
        dialog = ctk.CTkToplevel(parent)
        dialog.title("Téléchargement de la mise à jour")
        dialog.geometry("450x200")
        dialog.resizable(False, False)

        # Centrer
        dialog.transient(parent)
        dialog.grab_set()
        dialog.lift()
        dialog.focus_force()

        # Frame principale
        main_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Label statut
        status_label = ctk.CTkLabel(
            main_frame,
            text="Téléchargement en cours...",
            font=("Arial", 14)
        )
        status_label.pack(pady=(0, 20))

        # Barre de progression
        progress_bar = ctk.CTkProgressBar(main_frame, width=400)
        progress_bar.pack(pady=(0, 10))
        progress_bar.set(0)

        # Label pourcentage
        percent_label = ctk.CTkLabel(
            main_frame,
            text="0%",
            font=("Arial", 12)
        )
        percent_label.pack(pady=(0, 20))

        # Bouton annuler
        cancel_btn = ctk.CTkButton(
            main_frame,
            text="Annuler",
            command=lambda: updater.cancel_download(),
            fg_color="gray40",
            hover_color="gray30"
        )
        cancel_btn.pack()

        # Fonction de mise à jour de la progression
        def update_progress(percent):
            if dialog.winfo_exists():
                progress_bar.set(percent / 100)
                percent_label.configure(text=f"{percent}%")

        # Télécharger dans un thread séparé
        def download_thread():
            installer = updater.download_update(download_url, progress_callback=update_progress)
            result['installer_path'] = installer
            if dialog.winfo_exists():
                dialog.destroy()

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

        # Attendre la fermeture
        dialog.wait_window()

        return result['installer_path']

    except Exception as e:
        logger.error(f"Error showing download progress: {e}")
        # Télécharger sans dialogue de progression
        return updater.download_update(download_url)


def check_updates_on_startup(parent=None, auto_install: bool = False):
    """
    Check for updates at application startup.

    Args:
        parent: Optional parent window for dialogs
        auto_install: If True, automatically start download and install
    """
    def _handle_update(version_info):
        if version_info is None:
            return

        if parent:
            parent.after(0, lambda: _show_dialog(version_info))
        elif auto_install:
            _auto_update(version_info)

    def _show_dialog(version_info):
        if show_update_dialog_ctk(parent, version_info):
            _auto_update(version_info)

    def _auto_update(version_info):
        updater = AutoUpdater()
        updater.remote_version_info = version_info
        download_url = updater.get_download_url()
        if download_url:
            # Afficher le dialogue de progression
            installer = show_download_progress_dialog(parent, updater, download_url)
            if installer:
                updater.install_update(installer)

    check_for_updates_async(_handle_update)
