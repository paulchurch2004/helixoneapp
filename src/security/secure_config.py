import os
import json
import keyring
import subprocess
from cryptography.fernet import Fernet
from pathlib import Path
import base64
import hashlib

class SecureConfigManager:
    """Gestionnaire de configuration sécurisé pour HelixOne"""
    
    def __init__(self):
        self.app_name = "HelixOne"
        self.config_dir = Path.home() / ".helixone"
        self.config_dir.mkdir(exist_ok=True)
        
        # Génération/récupération de la clé de chiffrement
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Fichier de configuration local (non sensible)
        self.config_file = self.config_dir / "app_config.json"
        
    def _get_or_create_encryption_key(self):
        """Récupère ou crée une clé de chiffrement unique par machine"""
        try:
            # Essayer de récupérer la clé depuis le keyring système
            key_b64 = keyring.get_password(self.app_name, "encryption_key")
            if key_b64:
                return base64.b64decode(key_b64.encode())
        except Exception:
            pass
        
        # Créer une nouvelle clé basée sur l'ID machine
        machine_id = self._get_machine_id()
        key = Fernet.generate_key()
        
        try:
            # Stocker dans le keyring système
            keyring.set_password(self.app_name, "encryption_key", base64.b64encode(key).decode())
        except Exception:
            # Fallback : stocker dans un fichier caché
            key_file = self.config_dir / ".key"
            with open(key_file, "wb") as f:
                f.write(key)
            # Rendre le fichier caché sur Windows (utilise subprocess pour éviter injection)
            if os.name == 'nt':
                try:
                    subprocess.run(['attrib', '+h', str(key_file)], check=False, capture_output=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass  # Ignorer si attrib n'est pas disponible
        
        return key
    
    def _get_machine_id(self):
        """Génère un ID unique par machine"""
        import platform
        machine_info = f"{platform.node()}{platform.machine()}{platform.processor()}"
        return hashlib.sha256(machine_info.encode()).hexdigest()[:16]
    
    def store_api_key(self, service: str, api_key: str):
        """Stocke une clé API de manière chiffrée"""
        if not api_key or api_key.strip() == "":
            raise ValueError(f"Clé API vide pour {service}")
        
        encrypted_key = self.cipher.encrypt(api_key.encode())
        
        try:
            # Stocker dans le keyring système (préféré)
            keyring.set_password(self.app_name, f"api_{service}", base64.b64encode(encrypted_key).decode())
        except Exception:
            # Fallback : fichier chiffré
            keys_file = self.config_dir / "api_keys.enc"
            keys_data = {}
            
            if keys_file.exists():
                try:
                    with open(keys_file, "r") as f:
                        keys_data = json.load(f)
                except Exception:
                    keys_data = {}
            
            keys_data[service] = base64.b64encode(encrypted_key).decode()
            
            with open(keys_file, "w") as f:
                json.dump(keys_data, f)
            
            # Permissions restrictives
            os.chmod(keys_file, 0o600)
    
    def get_api_key(self, service: str) -> str:
        """Récupère une clé API déchiffrée"""
        encrypted_key_b64 = None
        
        try:
            # Essayer le keyring système
            encrypted_key_b64 = keyring.get_password(self.app_name, f"api_{service}")
        except Exception:
            pass
        
        if not encrypted_key_b64:
            # Fallback : fichier chiffré
            keys_file = self.config_dir / "api_keys.enc"
            if keys_file.exists():
                try:
                    with open(keys_file, "r") as f:
                        keys_data = json.load(f)
                    encrypted_key_b64 = keys_data.get(service)
                except Exception:
                    pass
        
        if not encrypted_key_b64:
            raise ValueError(f"Clé API non trouvée pour {service}")
        
        try:
            encrypted_key = base64.b64decode(encrypted_key_b64.encode())
            return self.cipher.decrypt(encrypted_key).decode()
        except Exception as e:
            raise ValueError(f"Impossible de déchiffrer la clé pour {service}: {e}")
    
    def list_configured_apis(self) -> list:
        """Liste les APIs configurées"""
        apis = []
        
        # Vérifier le keyring
        try:
            for service in ["openai", "finnhub", "polygon", "twelvedata", "alphavantage", "eod"]:
                if keyring.get_password(self.app_name, f"api_{service}"):
                    apis.append(service)
        except Exception:
            pass
        
        # Vérifier le fichier
        keys_file = self.config_dir / "api_keys.enc"
        if keys_file.exists():
            try:
                with open(keys_file, "r") as f:
                    keys_data = json.load(f)
                apis.extend(keys_data.keys())
            except Exception:
                pass
        
        return list(set(apis))
    
    def save_app_config(self, config: dict):
        """Sauvegarde la configuration application (non sensible)"""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def load_app_config(self) -> dict:
        """Charge la configuration application"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "theme": "dark",
            "language": "fr",
            "timeout": 30,
            "retries": 3,
            "debug": False
        }
    
    def migrate_from_old_config(self, old_config_path: str):
        """Migration depuis l'ancien config.json"""
        if not os.path.exists(old_config_path):
            return
        
        try:
            with open(old_config_path, "r") as f:
                old_config = json.load(f)
            
            # Migrer les clés API
            api_keys = {
                "openai": old_config.get("openai_api_key"),
                "finnhub": old_config.get("finnhub_api_key"),
                "polygon": old_config.get("polygon_api_key"),
                "twelvedata": old_config.get("twelve_data_api_key"),
                "alphavantage": old_config.get("alpha_vantage_api_key"),
                "eod": old_config.get("eod_api_key")
            }
            
            for service, key in api_keys.items():
                if key and key.strip():
                    self.store_api_key(service, key)
            
            # Migrer la config app
            app_config = old_config.get("app_settings", {})
            self.save_app_config(app_config)
            
            print(f"Migration réussie depuis {old_config_path}")
            print("IMPORTANT: Supprimez maintenant l'ancien fichier config.json !")
            
        except Exception as e:
            print(f"Erreur lors de la migration: {e}")


# Interface simple pour le reste du code
_config_manager = None

def get_config_manager():
    """Singleton du gestionnaire de config"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SecureConfigManager()
    return _config_manager

def get_api_key(service: str) -> str:
    """Interface simple pour récupérer une clé API"""
    return get_config_manager().get_api_key(service)

def setup_api_keys():
    """Assistant de configuration initial"""
    print("=== Configuration sécurisée des clés API HelixOne ===")
    manager = get_config_manager()
    
    api_services = {
        "openai": "OpenAI API Key",
        "finnhub": "Finnhub API Key", 
        "polygon": "Polygon API Key",
        "twelvedata": "Twelve Data API Key",
        "alphavantage": "Alpha Vantage API Key",
        "eod": "EOD Historical Data API Key"
    }
    
    for service, description in api_services.items():
        key = input(f"{description} (laissez vide pour ignorer): ").strip()
        if key:
            try:
                manager.store_api_key(service, key)
                print(f"✅ {service} configuré")
            except Exception as e:
                print(f"❌ Erreur {service}: {e}")
    
    print("\nConfiguration terminée. Les clés sont chiffrées et sécurisées.")

if __name__ == "__main__":
    # Migration automatique si ancien config.json existe
    manager = SecureConfigManager()
    if os.path.exists("config.json"):
        manager.migrate_from_old_config("config.json")
    else:
        setup_api_keys()