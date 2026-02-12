"""
Gestionnaire d'identifiant d'appareil pour HelixOne
Génère et stocke un identifiant unique pour chaque appareil
"""

import os
import uuid
import json
import platform
from pathlib import Path
from typing import Optional


class DeviceManager:
    """
    Gère l'identifiant unique de l'appareil pour la reconnaissance automatique
    """

    def __init__(self):
        """Initialiser le gestionnaire d'appareil"""
        # Fichier de stockage de l'ID d'appareil
        self.device_file = os.path.expanduser("~/.helixone_device.json")
        self._device_id: Optional[str] = None
        self._device_name: Optional[str] = None

    def get_device_id(self) -> str:
        """
        Obtenir l'identifiant unique de cet appareil

        Returns:
            ID unique de l'appareil (créé s'il n'existe pas)
        """
        if self._device_id:
            return self._device_id

        # Charger depuis le fichier si existe
        if os.path.exists(self.device_file):
            try:
                with open(self.device_file, 'r') as f:
                    data = json.load(f)
                    self._device_id = data.get('device_id')
                    self._device_name = data.get('device_name')

                    if self._device_id:
                        return self._device_id
            except Exception as e:
                print(f"Erreur lecture device_id: {e}")

        # Créer un nouveau device_id
        self._device_id = self._generate_device_id()
        self._device_name = self._get_device_name()
        self._save_device_info()

        return self._device_id

    def get_device_name(self) -> str:
        """
        Obtenir le nom de l'appareil

        Returns:
            Nom de l'appareil (ex: "MacBook Pro de Paul")
        """
        if self._device_name:
            return self._device_name

        # S'assurer que le device_id est chargé
        self.get_device_id()
        return self._device_name or "Appareil inconnu"

    def _generate_device_id(self) -> str:
        """
        Générer un identifiant unique pour cet appareil

        Combine:
        - MAC address (uuid.getnode())
        - UUID aléatoire
        - Hostname

        Returns:
            ID unique de l'appareil
        """
        # Récupérer la MAC address comme entier
        mac = uuid.getnode()

        # Générer un UUID aléatoire
        random_uuid = uuid.uuid4()

        # Combiner pour créer un ID unique
        # Format: helixone_{mac}_{uuid}
        device_id = f"helixone_{mac:012x}_{random_uuid.hex[:16]}"

        return device_id

    def _get_device_name(self) -> str:
        """
        Obtenir un nom lisible pour l'appareil

        Returns:
            Nom de l'appareil (ex: "MacBook Pro" ou "DESKTOP-ABC123")
        """
        try:
            # Sur macOS
            if platform.system() == 'Darwin':
                import subprocess
                result = subprocess.run(
                    ['scutil', '--get', 'ComputerName'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            # Sur Windows
            elif platform.system() == 'Windows':
                import os
                computer_name = os.environ.get('COMPUTERNAME')
                if computer_name:
                    return computer_name

            # Fallback sur hostname
            return platform.node() or "Appareil inconnu"

        except Exception:
            # Fallback générique
            system = platform.system()
            if system == 'Darwin':
                return "Mac"
            elif system == 'Windows':
                return "PC Windows"
            else:
                return "Appareil"

    def _save_device_info(self):
        """Sauvegarder les informations d'appareil"""
        if not self._device_id:
            return

        data = {
            'device_id': self._device_id,
            'device_name': self._device_name,
            'platform': platform.system(),
            'platform_version': platform.version(),
            'created_at': str(uuid.uuid1().time)
        }

        try:
            with open(self.device_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde device_info: {e}")

    def clear_device_info(self):
        """Supprimer les informations d'appareil (oublier cet appareil)"""
        self._device_id = None
        self._device_name = None

        if os.path.exists(self.device_file):
            try:
                os.remove(self.device_file)
            except Exception as e:
                print(f"Erreur suppression device_info: {e}")


# Instance globale
_device_manager = DeviceManager()


def get_device_id() -> str:
    """Raccourci pour obtenir le device_id"""
    return _device_manager.get_device_id()


def get_device_name() -> str:
    """Raccourci pour obtenir le nom de l'appareil"""
    return _device_manager.get_device_name()


# Test du module
if __name__ == "__main__":
    print("🧪 Test du DeviceManager\n")

    dm = DeviceManager()

    device_id = dm.get_device_id()
    device_name = dm.get_device_name()

    print(f"Device ID: {device_id}")
    print(f"Device Name: {device_name}")
    print(f"Platform: {platform.system()} {platform.version()}")
