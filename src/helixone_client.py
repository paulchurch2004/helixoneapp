"""
HelixOne API Client
Client Python pour communiquer avec le backend HelixOne

Usage:
    from helixone_client import HelixOneClient
    
    client = HelixOneClient()
    client.register("user@email.com", "password")
    client.login("user@email.com", "password")
    license = client.get_license_status()
"""

import os
import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime


class HelixOneAPIError(Exception):
    """Exception levée lors d'erreurs API"""
    pass


class HelixOneClient:
    """
    Client pour l'API HelixOne
    
    Attributes:
        base_url (str): URL de base de l'API
        token (str): Token JWT d'authentification
        user (dict): Informations de l'utilisateur connecté
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialiser le client
        
        Args:
            base_url: URL du backend API (par défaut: http://127.0.0.1:8000)
        """
        if base_url is None:
            from src.config import get_api_url
            base_url = get_api_url()
        self.base_url = base_url.rstrip('/')
        self.token: Optional[str] = None
        self.user: Optional[Dict[str, Any]] = None
        self.timeout = 30  # Timeout en secondes
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        require_auth: bool = False
    ) -> Dict[str, Any]:
        """
        Effectuer une requête HTTP
        
        Args:
            method: Méthode HTTP (GET, POST, etc.)
            endpoint: Endpoint de l'API (ex: /auth/login)
            data: Données à envoyer (optionnel)
            require_auth: Si True, ajoute le token d'authentification
        
        Returns:
            Réponse JSON de l'API
        
        Raises:
            HelixOneAPIError: En cas d'erreur API
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        # Ajouter le token si authentification requise
        if require_auth:
            if not self.token:
                raise HelixOneAPIError("Non authentifié. Appelez login() d'abord.")
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"Méthode HTTP non supportée: {method}")
            
            # Vérifier le statut de la réponse
            if response.status_code >= 400:
                error_msg = f"Erreur {response.status_code}"
                try:
                    error_detail = response.json().get("detail", response.text)
                    error_msg += f": {error_detail}"
                except Exception:
                    error_msg += f": {response.text}"
                raise HelixOneAPIError(error_msg)
            
            return response.json()
        
        except requests.exceptions.ConnectionError:
            raise HelixOneAPIError(
                f"Impossible de se connecter au serveur {self.base_url}. "
                "Vérifiez que le backend est lancé."
            )
        except requests.exceptions.Timeout:
            raise HelixOneAPIError(f"Timeout: le serveur ne répond pas après {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise HelixOneAPIError(f"Erreur réseau: {str(e)}")
    
    # ========================================================================
    # AUTHENTIFICATION
    # ========================================================================
    
    def register(
        self, 
        email: str, 
        password: str, 
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Créer un nouveau compte utilisateur
        
        Args:
            email: Adresse email
            password: Mot de passe
            first_name: Prénom (optionnel)
            last_name: Nom (optionnel)
        
        Returns:
            Dict contenant le token et les infos utilisateur
        
        Example:
            >>> client = HelixOneClient()
            >>> result = client.register("user@example.com", "MyPassword123!")
            >>> print(f"Compte créé: {result['user']['email']}")
        """
        data = {
            "email": email,
            "password": password,
            "first_name": first_name,
            "last_name": last_name
        }
        
        result = self._make_request("POST", "/auth/register", data)
        
        # Sauvegarder le token et les infos utilisateur
        self.token = result["access_token"]
        self.user = result["user"]
        
        return result
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Se connecter avec email et mot de passe
        
        Args:
            email: Adresse email
            password: Mot de passe
        
        Returns:
            Dict contenant le token et les infos utilisateur
        
        Example:
            >>> client = HelixOneClient()
            >>> result = client.login("user@example.com", "MyPassword123!")
            >>> print(f"Connecté en tant que: {result['user']['email']}")
        """
        data = {
            "email": email,
            "password": password
        }
        
        result = self._make_request("POST", "/auth/login", data)
        
        # Sauvegarder le token et les infos utilisateur
        self.token = result["access_token"]
        self.user = result["user"]
        
        return result
    
    def is_authenticated(self) -> bool:
        """
        Vérifier si l'utilisateur est authentifié
        
        Returns:
            True si un token existe
        """
        return self.token is not None
    
    def logout(self):
        """Déconnexion (supprime le token local)"""
        self.token = None
        self.user = None
    
    # ========================================================================
    # LICENCES
    # ========================================================================
    
    def get_license_status(self) -> Dict[str, Any]:
        """
        Récupérer le statut de la licence de l'utilisateur connecté
        
        Returns:
            Dict contenant les infos de licence
        
        Example:
            >>> client = HelixOneClient()
            >>> client.login("user@example.com", "password")
            >>> license = client.get_license_status()
            >>> print(f"Type: {license['license_type']}")
            >>> print(f"Jours restants: {license['days_remaining']}")
            >>> print(f"Quota analyses/jour: {license['quota_daily_analyses']}")
        """
        return self._make_request("GET", "/licenses/status", require_auth=True)
    
    def is_license_valid(self) -> bool:
        """
        Vérifier rapidement si la licence est valide
        
        Returns:
            True si la licence est active et non expirée
        """
        try:
            license = self.get_license_status()
            return (
                license['status'] == 'active' and 
                license['days_remaining'] > 0
            )
        except Exception:
            return False
    
    # ========================================================================
    # GESTION DU COMPTE
    # ========================================================================

    def change_password(self, current_password: str, new_password: str) -> Dict[str, Any]:
        """
        Changer le mot de passe de l'utilisateur connecté

        Args:
            current_password: Mot de passe actuel
            new_password: Nouveau mot de passe

        Returns:
            Dict avec success et message
        """
        data = {
            "current_password": current_password,
            "new_password": new_password
        }
        return self._make_request("POST", "/auth/change-password", data, require_auth=True)

    def delete_account(self, password: str, confirmation: str) -> Dict[str, Any]:
        """
        Supprimer définitivement le compte utilisateur

        Args:
            password: Mot de passe pour confirmer
            confirmation: Doit être "SUPPRIMER"

        Returns:
            Dict avec success et message
        """
        data = {
            "password": password,
            "confirmation": confirmation
        }
        return self._make_request("POST", "/auth/delete-account", data, require_auth=True)

    # ========================================================================
    # ANALYSES (à implémenter plus tard)
    # ========================================================================
    
    def analyze(self, ticker: str, mode: str = "Standard") -> Dict[str, Any]:
        """
        Analyser une action (TODO: route à créer dans le backend)
        
        Args:
            ticker: Symbole de l'action (ex: AAPL)
            mode: Mode d'analyse (Standard, Conservative, Aggressive)
        
        Returns:
            Résultats de l'analyse
        
        Example:
            >>> result = client.analyze("AAPL", mode="Standard")
            >>> print(f"Score: {result['score_final']}/100")
            >>> print(f"Recommandation: {result['recommendation']}")
        """
        data = {
            "ticker": ticker,
            "mode": mode
        }
        return self._make_request("POST", "/analyses/analyze", data, require_auth=True)
    
    # ========================================================================
    # UTILITAIRES
    # ========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Vérifier que l'API est accessible
        
        Returns:
            Infos de santé de l'API
        """
        return self._make_request("GET", "/health")
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtenir les informations de l'utilisateur connecté
        
        Returns:
            Dict avec les infos utilisateur ou None si non connecté
        """
        return self.user
    
    def __repr__(self):
        """Représentation du client"""
        auth_status = "authentifié" if self.is_authenticated() else "non authentifié"
        user_email = self.user['email'] if self.user else "N/A"
        return f"<HelixOneClient({auth_status}, user={user_email})>"


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """Démonstration d'utilisation du client"""
    
    print("=" * 70)
    print("🧪 DÉMONSTRATION DU CLIENT HELIXONE")
    print("=" * 70)
    
    # Créer le client
    client = HelixOneClient()
    
    # Test 1: Health check
    print("\n1️⃣  Health Check")
    try:
        health = client.health_check()
        print(f"✅ API accessible: {health['app_name']} v{health['version']}")
    except HelixOneAPIError as e:
        print(f"❌ {e}")
        exit(1)
    
    # Test 2: Inscription
    print("\n2️⃣  Inscription d'un nouvel utilisateur")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    test_email = f"demo_{timestamp}@helixone.com"
    # Mot de passe de démo - utiliser variable d'environnement en production
    demo_password = os.environ.get("HELIXONE_DEMO_PASSWORD", "Demo123456!")

    try:
        result = client.register(
            email=test_email,
            password=demo_password,
            first_name="Demo",
            last_name="User"
        )
        print(f"✅ Compte créé: {result['user']['email']}")
        print(f"   Token reçu: {result['access_token'][:50]}...")
    except HelixOneAPIError as e:
        print(f"❌ {e}")
    
    # Test 3: Vérifier la licence
    print("\n3️⃣  Vérification de la licence")
    try:
        license = client.get_license_status()
        print(f"✅ Licence: {license['license_key']}")
        print(f"   Type: {license['license_type']}")
        print(f"   Statut: {license['status']}")
        print(f"   Jours restants: {license['days_remaining']}")
        print(f"   Quota analyses: {license['quota_daily_analyses']}/jour")
    except HelixOneAPIError as e:
        print(f"❌ {e}")
    
    # Test 4: Déconnexion et reconnexion
    print("\n4️⃣  Déconnexion et reconnexion")
    client.logout()
    print(f"   Déconnecté: {not client.is_authenticated()}")
    
    try:
        result = client.login(test_email, "Demo123456!")
        print(f"✅ Reconnecté: {result['user']['email']}")
    except HelixOneAPIError as e:
        print(f"❌ {e}")
    
    print("\n" + "=" * 70)
    print("✅ Démonstration terminée!")
    print("=" * 70)
