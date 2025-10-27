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
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialiser le client
        
        Args:
            base_url: URL du backend API (par défaut: http://127.0.0.1:8000)
        """
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
                except:
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
        except:
            return False
    
    # ========================================================================
    # ANALYSES (à implémenter plus tard)
    # ========================================================================
    
    def analyze(self, ticker: str, mode: str = "Standard") -> Dict[str, Any]:
        """
        Analyser une action avec le moteur ML intelligent

        Combine :
        - Analyse FXI (5 dimensions)
        - Prédictions ML (1j, 3j, 7j)
        - Recommandations actionnables
        - Health Score global

        Args:
            ticker: Symbole de l'action (ex: AAPL)
            mode: Mode d'analyse (Standard, Conservative, Aggressive)

        Returns:
            Résultats de l'analyse complète avec ML

        Example:
            >>> result = client.analyze("AAPL", mode="Standard")
            >>> print(f"Health Score: {result['health_score']}/100")
            >>> print(f"Recommandation: {result['recommendation_final']}")
            >>> print(f"ML Signal: {result['ml_predictions']['signal']}")
            >>> print(f"Prédiction 7j: {result['ml_predictions']['prediction_7d']}")
        """
        data = {
            "ticker": ticker,
            "mode": mode
        }
        return self._make_request("POST", "/api/analysis/complete", data, require_auth=True)

    def deep_analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyser une action avec le système COMPLET 8 étapes (même analyse que 2x/jour)

        Cette analyse ultra-complète inclut:
        - ÉTAPE 1: Data collection (35+ sources: Reddit, StockTwits, News, Google Trends, FRED, etc.)
        - ÉTAPE 2: Sentiment analysis (trends, velocity, patterns)
        - ÉTAPE 3: Position analysis (correlations, diversification, health scores)
        - ÉTAPE 4: ML predictions (XGBoost + LSTM avec 120+ features)
        - ÉTAPE 5: Recommendations (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL détaillées)
        - ÉTAPE 6: Alerts (Critical/Important/Info/Opportunity)
        - ÉTAPE 7: Economic events (Fed, earnings, macro events à venir)
        - ÉTAPE 8: Executive summary

        C'est exactement la MÊME analyse que celle exécutée automatiquement
        à 7h00 et 17h00 EST sur tout le portfolio, mais appliquée à une seule action.

        Args:
            ticker: Symbole de l'action (ex: AAPL, MSFT, TSLA)

        Returns:
            Résultats ultra-complets avec toutes les étapes d'analyse

        Example:
            >>> result = client.deep_analyze("AAPL")
            >>> print(f"Health Score: {result['position_analysis']['health_score']}/100")
            >>> print(f"Recommandation: {result['recommendation']['action']}")
            >>> print(f"ML Signal: {result['ml_predictions']['signal']}")
            >>> print(f"Sentiment Trend: {result['sentiment_analysis']['trend']}")
            >>> print(f"Alertes Critiques: {len(result['alerts']['critical'])}")
            >>> print(f"Événements à venir: {len(result['upcoming_events'])}")
            >>> print(f"\\nRésumé Exécutif:\\n{result['executive_summary']}")
        """
        data = {
            "ticker": ticker
        }
        return self._make_request("POST", "/api/analysis/stock-deep-analysis", data, require_auth=True)

    def get_portfolio_analysis(self) -> Dict[str, Any]:
        """
        Récupérer la dernière analyse de portfolio

        Returns:
            Dernière analyse complète du portfolio (analyses 2x/jour)

        Example:
            >>> analysis = client.get_portfolio_analysis()
            >>> print(f"Health Score Portfolio: {analysis['health_score']}/100")
            >>> print(f"Alertes: {len(analysis['alerts'])}")
            >>> print(f"Recommandations: {len(analysis['recommendations'])}")
        """
        return self._make_request("GET", "/api/portfolio/analysis/latest", require_auth=True)

    def get_portfolio_alerts(self, severity: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupérer les alertes du portfolio

        Args:
            severity: Filtrer par sévérité (CRITICAL, WARNING, OPPORTUNITY, INFO)

        Returns:
            Liste des alertes actives
        """
        endpoint = "/api/portfolio/alerts"
        if severity:
            endpoint += f"?severity={severity}"
        return self._make_request("GET", endpoint, require_auth=True)

    def get_portfolio_recommendations(self) -> Dict[str, Any]:
        """
        Récupérer les recommandations du portfolio

        Returns:
            Liste des recommandations actives (BUY/HOLD/SELL)
        """
        return self._make_request("GET", "/api/portfolio/recommendations", require_auth=True)
    
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
    
    try:
        result = client.register(
            email=test_email,
            password="Demo123456!",
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
