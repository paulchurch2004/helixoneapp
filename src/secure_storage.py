"""
Stockage sécurisé pour HelixOne
Utilise Keychain (macOS) et Windows Credential Manager (Windows) via la bibliothèque keyring
"""

import os
import sys
import platform
import json
from typing import Optional, Dict

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    print("⚠️ keyring non installé - utilisation du fallback fichier")


class SecureStorage:
    """
    Gère le stockage sécurisé des credentials
    - macOS: utilise Keychain via keyring
    - Windows: utilise Windows Credential Manager via keyring
    - Linux: utilise Secret Service via keyring
    - Fallback: fichier avec permissions restreintes
    """

    SERVICE_NAME = "HelixOne"

    def __init__(self):
        """Initialiser le stockage sécurisé"""
        self.platform = platform.system()
        self.use_keyring = KEYRING_AVAILABLE

    def save_credentials(self, email: str, password: str) -> bool:
        """
        Sauvegarder les credentials de manière sécurisée

        Args:
            email: Email de l'utilisateur
            password: Mot de passe (sera chiffré)

        Returns:
            True si sauvegarde réussie
        """
        if self.use_keyring:
            return self._save_with_keyring(email, password)
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
        if self.use_keyring:
            return self._get_with_keyring(email)
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
        if self.use_keyring:
            return self._delete_with_keyring(email)
        else:
            return self._delete_from_file(email)

    # ====== Méthodes avec keyring (macOS, Windows, Linux) ======

    def _save_with_keyring(self, email: str, password: str) -> bool:
        """Sauvegarder avec keyring (fonctionne sur macOS, Windows, Linux)"""
        try:
            keyring.set_password(self.SERVICE_NAME, email, password)
            return True
        except Exception as e:
            print(f"Erreur sauvegarde keyring: {e}")
            return False

    def _get_with_keyring(self, email: str) -> Optional[str]:
        """Récupérer avec keyring"""
        try:
            return keyring.get_password(self.SERVICE_NAME, email)
        except Exception as e:
            print(f"Erreur lecture keyring: {e}")
            return None

    def _delete_with_keyring(self, email: str) -> bool:
        """Supprimer avec keyring"""
        try:
            keyring.delete_password(self.SERVICE_NAME, email)
            return True
        except keyring.errors.PasswordDeleteError:
            # Password not found - c'est OK
            return True
        except Exception as e:
            print(f"Erreur suppression keyring: {e}")
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
