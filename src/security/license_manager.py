"""
Système de licensing et authentification pour HelixOne
Gère les abonnements, essais gratuits, et limitations
"""

import os
import json
import hashlib
import hmac
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import platform
import uuid
from cryptography.fernet import Fernet
import base64

from src.utils.logger import get_logger

class LicenseManager:
    """Gestionnaire de licences pour HelixOne"""
    
    def __init__(self):
        self.logger = get_logger()
        self.license_dir = Path.home() / ".helixone"
        self.license_file = self.license_dir / "license.enc"
        
        # Configuration serveur (à modifier selon votre serveur)
        self.license_server = "https://api.helixone.com"  # Remplacez par votre API
        self.app_version = "1.0.0"
        
        # Types d'abonnements
        self.subscription_types = {
            "trial": {
                "name": "Essai gratuit",
                "duration_days": 14,
                "api_calls_per_day": 50,
                "features": ["basic_analysis", "dashboard", "charts"],
                "max_analyses_per_day": 10
            },
            "basic": {
                "name": "Basic",
                "duration_days": 30,
                "api_calls_per_day": 200,
                "features": ["basic_analysis", "dashboard", "charts", "exports"],
                "max_analyses_per_day": 50
            },
            "premium": {
                "name": "Premium", 
                "duration_days": 30,
                "api_calls_per_day": 1000,
                "features": ["basic_analysis", "dashboard", "charts", "exports", 
                           "advanced_analysis", "alerts", "formation"],
                "max_analyses_per_day": 200
            },
            "professional": {
                "name": "Professionnel",
                "duration_days": 30,
                "api_calls_per_day": -1,  # Illimité
                "features": ["all"],
                "max_analyses_per_day": -1  # Illimité
            }
        }
        
        self.current_license = self.load_license()
        self.usage_stats = self.load_usage_stats()
    
    def get_machine_id(self) -> str:
        """Génère un ID unique par machine"""
        try:
            # Utiliser plusieurs sources pour l'ID machine
            machine_info = [
                platform.node(),
                platform.machine(), 
                platform.processor(),
                str(uuid.getnode()),  # MAC address
            ]
            
            # Ajouter l'UUID du système si disponible (Windows)
            try:
                import subprocess
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ['wmic', 'csproduct', 'get', 'UUID'], 
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            machine_info.append(lines[1].strip())
            except Exception:
                pass
            
            combined = "|".join(filter(None, machine_info))
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
            
        except Exception as e:
            self.logger.log_error(e, "Machine ID generation", "Erreur génération ID machine")
            # Fallback avec timestamp
            return hashlib.sha256(f"fallback_{time.time()}".encode()).hexdigest()[:32]
    
    def create_trial_license(self) -> Dict[str, Any]:
        """Crée une licence d'essai"""
        machine_id = self.get_machine_id()
        
        trial_license = {
            "license_type": "trial",
            "machine_id": machine_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=14)).isoformat(),
            "user_email": "",
            "subscription": self.subscription_types["trial"],
            "status": "active",
            "version": self.app_version
        }
        
        self.current_license = trial_license
        self.save_license(trial_license)
        
        self.logger.log_user_action("trial_license_created", {
            "machine_id": machine_id[:8],  # Partial ID for privacy
            "expires_at": trial_license["expires_at"]
        })
        
        return trial_license
    
    def activate_license(self, license_key: str, user_email: str) -> Dict[str, Any]:
        """Active une licence avec une clé"""
        try:
            machine_id = self.get_machine_id()
            
            # Appel au serveur de licences
            response = self.validate_license_with_server(license_key, user_email, machine_id)
            
            if response["success"]:
                license_data = response["license"]
                license_data["machine_id"] = machine_id
                license_data["activated_at"] = datetime.now().isoformat()
                license_data["version"] = self.app_version
                
                self.current_license = license_data
                self.save_license(license_data)
                
                self.logger.log_user_action("license_activated", {
                    "license_type": license_data.get("license_type"),
                    "user_email": user_email,
                    "expires_at": license_data.get("expires_at")
                })
                
                return {"success": True, "license": license_data}
            else:
                return {"success": False, "error": response["error"]}
                
        except Exception as e:
            self.logger.log_error(e, "License activation", "Erreur activation licence")
            return {"success": False, "error": "Erreur de connexion au serveur"}
    
    def validate_license_with_server(self, license_key: str, email: str, machine_id: str) -> Dict:
        """Valide une licence avec le serveur"""
        try:
            # Simulation d'appel API (remplacez par votre vraie API)
            payload = {
                "license_key": license_key,
                "email": email,
                "machine_id": machine_id,
                "app_version": self.app_version,
                "platform": platform.system()
            }
            
            # Headers de sécurité
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"HelixOne/{self.app_version}",
                "X-App-Version": self.app_version
            }
            
            # Simulation de validation (en production, utilisez votre vraie API)
            if license_key.startswith("TRIAL"):
                return {
                    "success": True,
                    "license": {
                        "license_type": "trial",
                        "subscription": self.subscription_types["trial"],
                        "expires_at": (datetime.now() + timedelta(days=14)).isoformat(),
                        "status": "active"
                    }
                }
            elif license_key.startswith("BASIC"):
                return {
                    "success": True,
                    "license": {
                        "license_type": "basic",
                        "subscription": self.subscription_types["basic"],
                        "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
                        "status": "active"
                    }
                }
            elif license_key.startswith("PREMIUM"):
                return {
                    "success": True,
                    "license": {
                        "license_type": "premium", 
                        "subscription": self.subscription_types["premium"],
                        "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
                        "status": "active"
                    }
                }
            else:
                return {"success": False, "error": "Clé de licence invalide"}
            
            # Code pour vraie API (décommentez et adaptez)
            """
            response = requests.post(
                f"{self.license_server}/validate",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": "Serveur inaccessible"}
            """
            
        except requests.RequestException as e:
            self.logger.log_error(e, "License server validation", "Erreur serveur licences")
            return {"success": False, "error": "Impossible de contacter le serveur"}
        except Exception as e:
            self.logger.log_error(e, "License validation", "Erreur validation licence")
            return {"success": False, "error": "Erreur validation"}
    
    def is_license_valid(self) -> bool:
        """Vérifie si la licence actuelle est valide"""
        if not self.current_license:
            return False
        
        try:
            # Vérifier l'expiration
            expires_at = datetime.fromisoformat(self.current_license["expires_at"])
            if datetime.now() > expires_at:
                self.logger.log_user_action("license_expired", {
                    "license_type": self.current_license.get("license_type"),
                    "expired_at": self.current_license["expires_at"]
                })
                return False
            
            # Vérifier l'ID machine
            if self.current_license.get("machine_id") != self.get_machine_id():
                self.logger.log_user_action("license_machine_mismatch", {
                    "expected": self.current_license.get("machine_id", "")[:8],
                    "actual": self.get_machine_id()[:8]
                })
                return False
            
            # Vérifier le statut
            if self.current_license.get("status") != "active":
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "License validity check", "Erreur vérification licence")
            return False
    
    def get_license_info(self) -> Dict[str, Any]:
        """Retourne les informations de licence"""
        if not self.current_license:
            return {"status": "no_license"}
        
        try:
            expires_at = datetime.fromisoformat(self.current_license["expires_at"])
            days_left = (expires_at - datetime.now()).days
            
            return {
                "status": "valid" if self.is_license_valid() else "invalid",
                "license_type": self.current_license.get("license_type", "unknown"),
                "expires_at": self.current_license["expires_at"],
                "days_left": max(0, days_left),
                "subscription": self.current_license.get("subscription", {}),
                "user_email": self.current_license.get("user_email", ""),
                "features": self.current_license.get("subscription", {}).get("features", [])
            }
        except Exception as e:
            self.logger.log_error(e, "License info", "Erreur info licence")
            return {"status": "error"}
    
    def has_feature(self, feature: str) -> bool:
        """Vérifie si une fonctionnalité est disponible"""
        if not self.is_license_valid():
            return feature in ["basic_analysis", "dashboard"]  # Fonctionnalités minimales
        
        subscription = self.current_license.get("subscription", {})
        features = subscription.get("features", [])
        
        return "all" in features or feature in features
    
    def can_make_api_call(self) -> bool:
        """Vérifie si on peut faire un appel API"""
        if not self.is_license_valid():
            return False
        
        subscription = self.current_license.get("subscription", {})
        daily_limit = subscription.get("api_calls_per_day", 0)
        
        if daily_limit == -1:  # Illimité
            return True
        
        today = datetime.now().date().isoformat()
        api_calls_today = self.usage_stats.get("api_calls", {}).get(today, 0)
        
        return api_calls_today < daily_limit
    
    def can_make_analysis(self) -> bool:
        """Vérifie si on peut faire une analyse"""
        if not self.is_license_valid():
            return False
        
        subscription = self.current_license.get("subscription", {})
        daily_limit = subscription.get("max_analyses_per_day", 0)
        
        if daily_limit == -1:  # Illimité
            return True
        
        today = datetime.now().date().isoformat()
        analyses_today = self.usage_stats.get("analyses", {}).get(today, 0)
        
        return analyses_today < daily_limit
    
    def record_api_call(self):
        """Enregistre un appel API"""
        today = datetime.now().date().isoformat()
        
        if "api_calls" not in self.usage_stats:
            self.usage_stats["api_calls"] = {}
        
        if today not in self.usage_stats["api_calls"]:
            self.usage_stats["api_calls"][today] = 0
        
        self.usage_stats["api_calls"][today] += 1
        self.save_usage_stats()
        
        self.logger.log_user_action("api_call_recorded", {
            "date": today,
            "total_today": self.usage_stats["api_calls"][today]
        })
    
    def record_analysis(self):
        """Enregistre une analyse"""
        today = datetime.now().date().isoformat()
        
        if "analyses" not in self.usage_stats:
            self.usage_stats["analyses"] = {}
        
        if today not in self.usage_stats["analyses"]:
            self.usage_stats["analyses"][today] = 0
        
        self.usage_stats["analyses"][today] += 1
        self.save_usage_stats()
        
        self.logger.log_user_action("analysis_recorded", {
            "date": today,
            "total_today": self.usage_stats["analyses"][today]
        })
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'usage"""
        if not self.is_license_valid():
            return {}
        
        today = datetime.now().date().isoformat()
        subscription = self.current_license.get("subscription", {})
        
        return {
            "api_calls_today": self.usage_stats.get("api_calls", {}).get(today, 0),
            "api_calls_limit": subscription.get("api_calls_per_day", 0),
            "analyses_today": self.usage_stats.get("analyses", {}).get(today, 0),
            "analyses_limit": subscription.get("max_analyses_per_day", 0),
            "license_type": self.current_license.get("license_type", "unknown")
        }
    
    def save_license(self, license_data: Dict[str, Any]):
        """Sauvegarde la licence de manière chiffrée"""
        try:
            # Chiffrer les données de licence
            key = Fernet.generate_key()
            cipher = Fernet(key)
            
            license_json = json.dumps(license_data, ensure_ascii=False)
            encrypted_data = cipher.encrypt(license_json.encode())
            
            # Stocker la clé et les données
            with open(self.license_file, "wb") as f:
                f.write(key + b"|||" + encrypted_data)
            
            # Permissions restrictives
            os.chmod(self.license_file, 0o600)
            
        except Exception as e:
            self.logger.log_error(e, "Save license", "Erreur sauvegarde licence")
    
    def load_license(self) -> Optional[Dict[str, Any]]:
        """Charge la licence depuis le fichier chiffré"""
        try:
            if not self.license_file.exists():
                return None
            
            with open(self.license_file, "rb") as f:
                data = f.read()
            
            if b"|||" not in data:
                return None
            
            key_data, encrypted_data = data.split(b"|||", 1)
            cipher = Fernet(key_data)
            
            decrypted_data = cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
            
        except Exception as e:
            self.logger.log_error(e, "Load license", "Erreur chargement licence")
            return None
    
    def load_usage_stats(self) -> Dict[str, Any]:
        """Charge les statistiques d'usage"""
        try:
            stats_file = self.license_dir / "usage_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.log_error(e, "Load usage stats", "Erreur chargement stats")
        
        return {}
    
    def save_usage_stats(self):
        """Sauvegarde les statistiques d'usage"""
        try:
            stats_file = self.license_dir / "usage_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.usage_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.log_error(e, "Save usage stats", "Erreur sauvegarde stats")
    
    def cleanup_old_stats(self):
        """Nettoie les anciennes statistiques (garde 30 jours)"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=30)).date().isoformat()
            
            for stat_type in ["api_calls", "analyses"]:
                if stat_type in self.usage_stats:
                    old_dates = [date for date in self.usage_stats[stat_type].keys() 
                               if date < cutoff_date]
                    for date in old_dates:
                        del self.usage_stats[stat_type][date]
            
            self.save_usage_stats()
            
        except Exception as e:
            self.logger.log_error(e, "Cleanup stats", "Erreur nettoyage stats")


# Instance globale
_license_manager = None

def get_license_manager() -> LicenseManager:
    """Singleton du gestionnaire de licences"""
    global _license_manager
    if _license_manager is None:
        _license_manager = LicenseManager()
    return _license_manager

# Interface simple pour le reste du code
def is_feature_available(feature: str) -> bool:
    """Vérifie si une fonctionnalité est disponible"""
    return get_license_manager().has_feature(feature)

def can_make_api_call() -> bool:
    """Vérifie si on peut faire un appel API"""
    return get_license_manager().can_make_api_call()

def can_make_analysis() -> bool:
    """Vérifie si on peut faire une analyse"""
    return get_license_manager().can_make_analysis()

def record_api_usage():
    """Enregistre l'usage d'une API"""
    get_license_manager().record_api_call()

def record_analysis_usage():
    """Enregistre l'usage d'une analyse"""
    get_license_manager().record_analysis()

# Décorateur pour protéger les fonctions
def require_license(feature: str = None):
    """Décorateur pour protéger les fonctions selon la licence"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_license_manager()
            
            if not manager.is_license_valid():
                raise PermissionError("Licence invalide ou expirée")
            
            if feature and not manager.has_feature(feature):
                raise PermissionError(f"Fonctionnalité '{feature}' non disponible dans votre abonnement")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_api_quota():
    """Décorateur pour vérifier le quota API"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_license_manager()
            
            if not manager.can_make_api_call():
                raise PermissionError("Quota d'appels API dépassé pour aujourd'hui")
            
            result = func(*args, **kwargs)
            manager.record_api_call()
            return result
        return wrapper
    return decorator

def require_analysis_quota():
    """Décorateur pour vérifier le quota d'analyses"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_license_manager()
            
            if not manager.can_make_analysis():
                raise PermissionError("Quota d'analyses dépassé pour aujourd'hui")
            
            result = func(*args, **kwargs)
            manager.record_analysis()
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test du système de licensing
    manager = LicenseManager()
    
    print("=== Test système de licensing HelixOne ===")
    
    # Test création licence d'essai
    trial = manager.create_trial_license()
    print(f"Licence d'essai créée: {trial['license_type']}")
    
    # Test validation
    print(f"Licence valide: {manager.is_license_valid()}")
    
    # Test fonctionnalités
    print(f"Dashboard disponible: {manager.has_feature('dashboard')}")
    print(f"Formation disponible: {manager.has_feature('formation')}")
    
    # Test quotas
    print(f"Peut faire API: {manager.can_make_api_call()}")
    print(f"Peut faire analyse: {manager.can_make_analysis()}")
    
    # Test stats
    manager.record_api_call()
    manager.record_analysis()
    stats = manager.get_usage_stats()
    print(f"Stats: {stats}")