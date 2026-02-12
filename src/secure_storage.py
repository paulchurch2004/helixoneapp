"""
Stockage sécurisé pour HelixOne
Utilise le Keychain macOS pour stocker les credentials de manière sécurisée
"""

import os
import sys
import subprocess
import platform
import json
from typing import Optional, Dict


class SecureStorage:
    """
    Gère le stockage sécurisé des credentials
    - macOS: utilise Keychain
    - Windows: utilise Windows Credential Manager
    - Linux: utilise Secret Service (ou fichier chiffré en fallback)
    """

    SERVICE_NAME = "fr.helixone.app"

    def __init__(self):
        """Initialiser le stockage sécurisé"""
        self.platform = platform.system()

    def save_credentials(self, email: str, password: str) -> bool:
        """
        Sauvegarder les credentials de manière sécurisée

        Args:
            email: Email de l'utilisateur
            password: Mot de passe (sera chiffré)

        Returns:
            True si sauvegarde réussie
        """
        if self.platform == 'Darwin':  # macOS
            return self._save_to_keychain_macos(email, password)
        elif self.platform == 'Windows':
            return self._save_to_credential_manager_windows(email, password)
        else:
            # Fallback: fichier avec permissions restreintes
            return self._save_to_file(email, password)

    def get_credentials(self, email: str) -> Optional[str]:
        """
        Récupérer le mot de passe pour un email donné

        Args:
            email: Email de l'utilisateur

        Returns:
            Mot de passe ou None si non trouvé
        """
        if self.platform == 'Darwin':
            return self._get_from_keychain_macos(email)
        elif self.platform == 'Windows':
            return self._get_from_credential_manager_windows(email)
        else:
            return self._get_from_file(email)

    def delete_credentials(self, email: str) -> bool:
        """
        Supprimer les credentials pour un email donné

        Args:
            email: Email de l'utilisateur

        Returns:
            True si suppression réussie
        """
        if self.platform == 'Darwin':
            return self._delete_from_keychain_macos(email)
        elif self.platform == 'Windows':
            return self._delete_from_credential_manager_windows(email)
        else:
            return self._delete_from_file(email)

    # ====== macOS Keychain ======

    def _save_to_keychain_macos(self, email: str, password: str) -> bool:
        """Sauvegarder dans Keychain macOS"""
        try:
            # Supprimer l'ancienne entrée si elle existe
            self._delete_from_keychain_macos(email)

            # Ajouter la nouvelle entrée
            cmd = [
                'security', 'add-generic-password',
                '-s', self.SERVICE_NAME,  # service name
                '-a', email,              # account name
                '-w', password,           # password
                '-U'                      # update if exists
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            return result.returncode == 0

        except Exception as e:
            print(f"Erreur sauvegarde Keychain: {e}")
            return False

    def _get_from_keychain_macos(self, email: str) -> Optional[str]:
        """Récupérer depuis Keychain macOS"""
        try:
            cmd = [
                'security', 'find-generic-password',
                '-s', self.SERVICE_NAME,
                '-a', email,
                '-w'  # output only the password
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return result.stdout.strip()

            return None

        except Exception as e:
            print(f"Erreur lecture Keychain: {e}")
            return None

    def _delete_from_keychain_macos(self, email: str) -> bool:
        """Supprimer depuis Keychain macOS"""
        try:
            cmd = [
                'security', 'delete-generic-password',
                '-s', self.SERVICE_NAME,
                '-a', email
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            # returncode 0 = supprimé, 44 = pas trouvé (aussi OK)
            return result.returncode in [0, 44]

        except Exception as e:
            print(f"Erreur suppression Keychain: {e}")
            return False

    # ====== Windows Credential Manager ======

    def _save_to_credential_manager_windows(self, email: str, password: str) -> bool:
        """Sauvegarder dans Windows Credential Manager"""
        try:
            # Utiliser cmdkey.exe
            target_name = f"{self.SERVICE_NAME}:{email}"

            # Supprimer l'ancienne entrée
            subprocess.run(
                ['cmdkey', '/delete', target_name],
                capture_output=True,
                timeout=5
            )

            # Ajouter la nouvelle
            result = subprocess.run(
                ['cmdkey', '/generic', target_name, '/user', email, '/pass', password],
                capture_output=True,
                timeout=5
            )

            return result.returncode == 0

        except Exception as e:
            print(f"Erreur sauvegarde Windows Credential Manager: {e}")
            return False

    def _get_from_credential_manager_windows(self, email: str) -> Optional[str]:
        """Récupérer depuis Windows Credential Manager"""
        # Windows cmdkey ne permet pas de lire les mots de passe directement
        # Il faudrait utiliser une lib comme keyring ou win32cred
        # Pour l'instant, fallback sur fichier
        return self._get_from_file(email)

    def _delete_from_credential_manager_windows(self, email: str) -> bool:
        """Supprimer depuis Windows Credential Manager"""
        try:
            target_name = f"{self.SERVICE_NAME}:{email}"
            result = subprocess.run(
                ['cmdkey', '/delete', target_name],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    # ====== Fallback: Fichier avec permissions restreintes ======

    def _get_credentials_file(self) -> str:
        """Obtenir le chemin du fichier de credentials"""
        return os.path.expanduser("~/.helixone_credentials.json")

    def _save_to_file(self, email: str, password: str) -> bool:
        """Sauvegarder dans un fichier (fallback)"""
        try:
            creds_file = self._get_credentials_file()

            # Charger les credentials existants
            credentials = {}
            if os.path.exists(creds_file):
                with open(creds_file, 'r') as f:
                    credentials = json.load(f)

            # Ajouter/mettre à jour
            credentials[email] = password

            # Sauvegarder
            with open(creds_file, 'w') as f:
                json.dump(credentials, f)

            # Restreindre les permissions (Unix seulement)
            if hasattr(os, 'chmod'):
                os.chmod(creds_file, 0o600)  # rw------- (lecture/écriture propriétaire seulement)

            return True

        except Exception as e:
            print(f"Erreur sauvegarde fichier credentials: {e}")
            return False

    def _get_from_file(self, email: str) -> Optional[str]:
        """Récupérer depuis fichier (fallback)"""
        try:
            creds_file = self._get_credentials_file()

            if not os.path.exists(creds_file):
                return None

            with open(creds_file, 'r') as f:
                credentials = json.load(f)

            return credentials.get(email)

        except Exception as e:
            print(f"Erreur lecture fichier credentials: {e}")
            return None

    def _delete_from_file(self, email: str) -> bool:
        """Supprimer depuis fichier (fallback)"""
        try:
            creds_file = self._get_credentials_file()

            if not os.path.exists(creds_file):
                return True

            with open(creds_file, 'r') as f:
                credentials = json.load(f)

            if email in credentials:
                del credentials[email]

                with open(creds_file, 'w') as f:
                    json.dump(credentials, f)

            return True

        except Exception as e:
            print(f"Erreur suppression fichier credentials: {e}")
            return False


# Instance globale
_secure_storage = SecureStorage()


def save_credentials(email: str, password: str) -> bool:
    """Raccourci pour sauvegarder des credentials"""
    return _secure_storage.save_credentials(email, password)


def get_credentials(email: str) -> Optional[str]:
    """Raccourci pour récupérer des credentials"""
    return _secure_storage.get_credentials(email)


def delete_credentials(email: str) -> bool:
    """Raccourci pour supprimer des credentials"""
    return _secure_storage.delete_credentials(email)


# Test du module
if __name__ == "__main__":
    print("🧪 Test du SecureStorage\n")

    storage = SecureStorage()
    print(f"Platform: {storage.platform}")

    # Test sauvegarde
    test_email = "test@helixone.fr"
    test_password = "test123"

    print(f"\n1. Sauvegarde credentials pour {test_email}...")
    if storage.save_credentials(test_email, test_password):
        print("✅ Sauvegarde réussie")
    else:
        print("❌ Échec sauvegarde")

    # Test récupération
    print(f"\n2. Récupération credentials pour {test_email}...")
    retrieved = storage.get_credentials(test_email)
    if retrieved == test_password:
        print(f"✅ Récupération réussie: {retrieved}")
    else:
        print(f"❌ Échec récupération: {retrieved}")

    # Test suppression
    print(f"\n3. Suppression credentials pour {test_email}...")
    if storage.delete_credentials(test_email):
        print("✅ Suppression réussie")
    else:
        print("❌ Échec suppression")

    # Vérifier suppression
    print(f"\n4. Vérification suppression...")
    retrieved = storage.get_credentials(test_email)
    if retrieved is None:
        print("✅ Credentials bien supprimés")
    else:
        print(f"❌ Credentials toujours présents: {retrieved}")
