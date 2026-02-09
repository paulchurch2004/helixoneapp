"""
Gestionnaire d'authentification pour HelixOne
Gère la session utilisateur et la sauvegarde du token JWT
"""

import os
import json
import sys
from typing import Optional, Dict, Any

# Ajouter le dossier parent au path pour trouver helixone_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helixone_client import HelixOneClient, HelixOneAPIError


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
    
    def is_logged_in(self) -> bool:
        """
        Vérifier si un utilisateur est connecté
        
        Returns:
            True si une session existe et est valide
        """
        # Charger la session sauvegardée
        if self.load_session():
            # Vérifier que le token est toujours valide en essayant d'accéder à la licence
            try:
                self.client.get_license_status()
                return True
            except HelixOneAPIError:
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
