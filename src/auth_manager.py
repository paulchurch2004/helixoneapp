"""
Gestionnaire d'authentification pour HelixOne
Gère la session utilisateur et la sauvegarde du token JWT
Supporte la connexion rapide et l'authentification biométrique
"""

import os
import json
import sys
from typing import Optional, Dict, Any, Callable

# Ajouter le dossier parent au path pour trouver helixone_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helixone_client import HelixOneClient, HelixOneAPIError
from src.secure_storage import SecureStorage
from src.biometric_auth import BiometricAuth
from src.device_manager import DeviceManager


class AuthManager:
    """
    Gère l'authentification et la session utilisateur
    
    Attributes:
        client (HelixOneClient): Client API
        token_file (str): Chemin du fichier de sauvegarde du token
    """
    
    def __init__(self, backend_url: str = None):
        """
        Initialiser le gestionnaire d'authentification

        Args:
            backend_url: URL du backend API
        """
        from src.config import get_api_url
        if backend_url is None:
            backend_url = get_api_url()
        self.client = HelixOneClient(base_url=backend_url)
        self.token_file = os.path.expanduser("~/.helixone_session.json")

        # Modules pour connexion rapide
        self.secure_storage = SecureStorage()
        self.biometric_auth = BiometricAuth()
        self.device_manager = DeviceManager()

        # Fichier pour stocker les préférences de connexion rapide
        self.quick_login_file = os.path.expanduser("~/.helixone_quick_login.json")
    
    def is_logged_in(self) -> bool:
        """
        Vérifier si un utilisateur est connecté
        
        Returns:
            True si une session existe et est valide
        """
        # Charger la session sauvegardée
        if self.load_session():
            # Vérifier que le token est toujours valide et rafraîchir les infos user
            try:
                self.client.fetch_current_user()
                self.save_session()
                return True
            except Exception:
                # Token invalide ou expiré
                self.clear_session()
                return False
        return False
    
    def load_session(self) -> bool:
        """
        Charger la session sauvegardée depuis le fichier
        
        Returns:
            True si la session a été chargée avec succès
        """
        if not os.path.exists(self.token_file):
            return False
        
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
                self.client.token = data.get('token')
                self.client.user = data.get('user')
                return True
        except Exception as e:
            print(f"Erreur chargement session: {e}")
            return False
    
    def save_session(self):
        """Sauvegarder la session actuelle dans un fichier"""
        if not self.client.token:
            return
        
        data = {
            'token': self.client.token,
            'user': self.client.user
        }
        
        try:
            with open(self.token_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde session: {e}")
    
    def clear_session(self):
        """Supprimer la session (déconnexion)"""
        self.client.logout()
        
        if os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
            except Exception as e:
                print(f"Erreur suppression session: {e}")
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Connecter un utilisateur
        
        Args:
            email: Email de l'utilisateur
            password: Mot de passe
        
        Returns:
            Résultat de la connexion
        
        Raises:
            HelixOneAPIError: En cas d'erreur
        """
        result = self.client.login(email, password)
        self.save_session()
        return result
    
    def register(
        self, 
        email: str, 
        password: str, 
        first_name: str = None,
        last_name: str = None
    ) -> Dict[str, Any]:
        """
        Créer un nouveau compte
        
        Args:
            email: Email
            password: Mot de passe
            first_name: Prénom
            last_name: Nom
        
        Returns:
            Résultat de l'inscription
        
        Raises:
            HelixOneAPIError: En cas d'erreur
        """
        result = self.client.register(email, password, first_name, last_name)
        self.save_session()
        return result
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Obtenir les informations de l'utilisateur connecté
        
        Returns:
            Dict avec les infos utilisateur ou None
        """
        return self.client.get_user_info()
    
    def get_license_info(self) -> Dict[str, Any]:
        """
        Obtenir les informations de la licence
        
        Returns:
            Dict avec les infos de licence
        
        Raises:
            HelixOneAPIError: En cas d'erreur
        """
        return self.client.get_license_status()
    
    def is_license_valid(self) -> bool:
        """
        Vérifier rapidement si la licence est valide

        Returns:
            True si la licence est active
        """
        try:
            return self.client.is_license_valid()
        except Exception:
            return False

    def logout(self):
        """
        Déconnexion de l'utilisateur (supprime le token et la session locale)
        """
        self.clear_session()

    # ====== Connexion rapide et biométrique ======

    def is_quick_login_enabled(self) -> bool:
        """
        Vérifier si la connexion rapide est activée

        Returns:
            True si un email est configuré pour connexion rapide
        """
        try:
            if os.path.exists(self.quick_login_file):
                with open(self.quick_login_file, 'r') as f:
                    data = json.load(f)
                    return data.get('enabled', False) and data.get('email') is not None
        except Exception:
            pass
        return False

    def get_quick_login_email(self) -> Optional[str]:
        """
        Obtenir l'email configuré pour connexion rapide

        Returns:
            Email ou None
        """
        try:
            if os.path.exists(self.quick_login_file):
                with open(self.quick_login_file, 'r') as f:
                    data = json.load(f)
                    if data.get('enabled'):
                        return data.get('email')
        except Exception:
            pass
        return None

    def enable_quick_login(self, email: str, password: str) -> bool:
        """
        Activer la connexion rapide pour cet appareil

        Args:
            email: Email de l'utilisateur
            password: Mot de passe (sera stocké de manière sécurisée)

        Returns:
            True si activation réussie
        """
        try:
            # Sauvegarder les credentials de manière sécurisée
            if not self.secure_storage.save_credentials(email, password):
                return False

            # Sauvegarder les préférences
            device_id = self.device_manager.get_device_id()
            device_name = self.device_manager.get_device_name()

            data = {
                'enabled': True,
                'email': email,
                'device_id': device_id,
                'device_name': device_name
            }

            with open(self.quick_login_file, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Erreur activation connexion rapide: {e}")
            return False

    def disable_quick_login(self):
        """Désactiver la connexion rapide et supprimer les credentials"""
        email = self.get_quick_login_email()

        if email:
            # Supprimer les credentials stockés
            self.secure_storage.delete_credentials(email)

        # Supprimer le fichier de préférences
        if os.path.exists(self.quick_login_file):
            try:
                os.remove(self.quick_login_file)
            except Exception as e:
                print(f"Erreur désactivation connexion rapide: {e}")

    def quick_login(self) -> bool:
        """
        Se connecter rapidement avec les credentials stockés

        Returns:
            True si connexion réussie
        """
        try:
            email = self.get_quick_login_email()
            if not email:
                return False

            # Récupérer le mot de passe stocké
            password = self.secure_storage.get_credentials(email)
            if not password:
                return False

            # Se connecter
            self.login(email, password)
            return True

        except Exception as e:
            print(f"Erreur connexion rapide: {e}")
            return False

    def is_biometric_available(self) -> bool:
        """
        Vérifier si l'authentification biométrique est disponible

        Returns:
            True si Touch ID/Face ID est disponible
        """
        return self.biometric_auth.is_available()

    def get_biometry_type(self) -> str:
        """
        Obtenir le type de biométrie disponible

        Returns:
            "touchid", "faceid", "none"
        """
        return self.biometric_auth.get_biometry_type()

    def biometric_login(
        self,
        callback: Optional[Callable[[bool, Optional[str]], None]] = None
    ) -> bool:
        """
        Se connecter avec l'authentification biométrique

        Args:
            callback: Fonction appelée avec (success, error_message)

        Returns:
            True si authentification lancée (résultat dans callback)
        """
        # Vérifier que la connexion rapide est activée
        email = self.get_quick_login_email()
        if not email:
            if callback:
                callback(False, "Connexion rapide non activée")
            return False

        # Demander l'authentification biométrique
        def on_biometric_result(success, error):
            if success:
                # Biométrie OK, se connecter
                try:
                    if self.quick_login():
                        if callback:
                            callback(True, None)
                    else:
                        if callback:
                            callback(False, "Échec connexion")
                except Exception as e:
                    if callback:
                        callback(False, str(e))
            else:
                if callback:
                    callback(False, error or "Authentification annulée")

        biometry_type = self.get_biometry_type()
        if biometry_type == "touchid":
            reason = "Utilisez Touch ID pour vous connecter à HelixOne"
        elif biometry_type == "faceid":
            reason = "Utilisez Face ID pour vous connecter à HelixOne"
        else:
            reason = "Authentifiez-vous pour accéder à HelixOne"

        return self.biometric_auth.authenticate(reason, on_biometric_result)


# Test du module
if __name__ == "__main__":
    print("🧪 Test du AuthManager\n")
    
    auth = AuthManager()
    
    # Test 1: Vérifier si connecté
    print(f"Connecté: {auth.is_logged_in()}")
    
    if auth.is_logged_in():
        user = auth.get_current_user()
        print(f"Utilisateur: {user['email']}")
        
        license = auth.get_license_info()
        print(f"Licence: {license['license_type']}")
        print(f"Jours restants: {license['days_remaining']}")
