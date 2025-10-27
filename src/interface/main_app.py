from PIL import Image
# from community_chat import CommunityChat as CommunityChat_Real
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
# from functools import lru_cache
from src.interface.community_chat import CommunityChat as CommunityChat_Real
from openai import OpenAI
from pathlib import Path
from rapidfuzz import fuzz, process
from src.data.tickers_fallback import SEARCH_INDEX
from src.data.tickers_fallback import SEARCH_INDEX as EXT_INDEX
from src.data.tickers_fallback import find_ticker as find_ticker_func
from src.data.tickers_fallback import get_ticker_suggestions as get_suggestions_func
from src.fxi_engine import get_analysis
from src.interface.chart_component import ChartPanel
from src.interface.formation_commerciale import afficher_formation_commerciale
from src.interface.ibkr_panel import IBKRPortfolioPanel
from src.interface.ml_results_display import MLResultsDisplay
from src.interface.portfolio_analysis_panel import PortfolioAnalysisPanel
from src.rapport import generer_rapport_v3
from tkinter import messagebox, filedialog
from typing import Dict, List, Optional, Any, Union, Type
import customtkinter as ctk
import importlib
import json
import logging
import openai
import os
import queue
import requests
import sys
import threading
import time
import tkinter as tk
import yfinance as yf

# Import du client API pour le nouveau moteur ML
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helixone_client import HelixOneClient

# === DECLARATION EXPLICITE POUR PYLANCE ===
CommunityChat: Type[ctk.CTkFrame]


# === CLASSE DE FALLBACK ===
class CommunityChat_Fallback(ctk.CTkFrame):
    def __init__(self, parent, user_profile=None):
        super().__init__(parent, fg_color="#1c1f26")

        error_frame = ctk.CTkFrame(self, fg_color="#2a2d36")
        error_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            error_frame,
            text="Chat Communautaire Non Disponible",
            font=("Arial", 18, "bold"),
            text_color="#FFA500"
        ).pack(pady=30)

        ctk.CTkLabel(
            error_frame,
            text="Le module community_chat.py est introuvable.\nVérifiez qu'il est présent dans interface/",
            font=("Arial", 14),
            text_color="#CCCCCC"
        ).pack(pady=20)


# === CHARGEMENT DU MODULE CHAT ===
def initialize_chat_module():
    """Initialise le module de chat"""
    global CommunityChat

    import_paths = ["interface.community_chat", "community_chat"]

    for module_path in import_paths:
        try:
            module = importlib.import_module(module_path)
            chat_class = getattr(module, "CommunityChat", None)

            if chat_class is not None:
                CommunityChat = chat_class
                print(f"✅ CommunityChat chargé depuis {module_path}")
                return True

        except ImportError:
            continue
        except Exception as e:
            print(f"Erreur avec {module_path}: {e}")
            continue

    CommunityChat = CommunityChat_Fallback
    print("⚠ Utilisation du fallback pour CommunityChat")
    return False


# Initialiser le module de chat
CHAT_AVAILABLE = initialize_chat_module()


# === IMPORT FALLBACK POUR MODULES MANQUANTS ===
try:
    pass
except ImportError as e:
    print(f"Erreur import modules locaux: {e}")
    
    def get_analysis(ticker, mode):
        return {"score_fxi": 75, "status": "simulation"}


# ============================================================================
# SYSTEME DE LOGGING - CRÉER EN PREMIER
# ============================================================================
def setup_logging() -> logging.Logger:
    """Configure le système de logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(
        log_dir / 'helixone.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s - %(message)s'))
    console_handler.setLevel(logging.WARNING)

    logger = logging.getLogger('helixone')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    
    
    
    logger.addHandler(console_handler)

    return logger


# CRÉER LE LOGGER IMMÉDIATEMENT
logger = setup_logging()


# ============================================================================
# IMPORTS DES EFFETS VISUELS - APRÈS LA CRÉATION DU LOGGER
# ============================================================================
# Importer les modules qui existent (glassmorphism, chart_animations, theme_manager)
# Les modules manquants (premium_effects, animated_widgets) auront des fallbacks
PREMIUM_EFFECTS_AVAILABLE = False

try:
    # Importer les modules qui existent réellement
    from src.interface.glassmorphism import (
        GlassFrame, GlassPanel, GlassCard,
        GlassButton, FloatingPanel
    )
    from src.interface.chart_animations import (
        AnimatedBarChart, AnimatedLineChart,
        ProgressBar, DonutChart, SparklineChart
    )
    from src.interface.theme_manager import theme_manager, AnimatedThemeTransition

    # Créer des classes fallback pour les modules manquants
    # (premium_effects et animated_widgets n'existent pas)
    class ParticleCanvas:
        def __init__(self, *args, **kwargs): pass
        def create(self): return None
        def stop(self): pass

    class WaveBackground:
        pass

    class GlowEffect:
        pass

    class RippleEffect:
        @staticmethod
        def create_ripple(*args): pass

    class AnimatedScore:
        def __init__(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def animate_to(self, *args): pass

    class CircularGauge:
        def __init__(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def set_value(self, *args, **kwargs): pass

    class RadarChart:
        def __init__(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def set_values(self, *args, **kwargs): pass

    class TypewriterLabel(ctk.CTkLabel):
        def __init__(self, parent, full_text="", *args, **kwargs):
            kwargs.pop('speed', None)
            super().__init__(parent, text=full_text, *args, **kwargs)
        def start_typing(self): pass

    class PulsingButton(ctk.CTkButton):
        def start_pulse(self): pass

    PREMIUM_EFFECTS_AVAILABLE = True
    logger.info("✨ Effets visuels chargés (avec fallbacks pour modules manquants)")

except ImportError as e:
    PREMIUM_EFFECTS_AVAILABLE = False
    logger.warning(f"⚠ Erreur lors du chargement des modules visuels: {e}")
    logger.info("L'application fonctionnera en mode fallback complet.")
    
    # Classes factices pour éviter les erreurs
    class ParticleCanvas:
        def __init__(self, *args, **kwargs): pass
        def create(self): return None
        def stop(self): pass
    
    class AnimatedScore:
        def __init__(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def animate_to(self, *args): pass
    
    class CircularGauge:
        def __init__(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def set_value(self, *args, **kwargs): pass
    
    class RadarChart:
        def __init__(self, *args, **kwargs): pass
        def pack(self, *args, **kwargs): pass
        def set_values(self, *args, **kwargs): pass
    
    class GlassFrame(ctk.CTkFrame):
        def __init__(self, parent, *args, **kwargs):
            kwargs.pop('border_glow', None)
            kwargs.pop('elevation', None)
            super().__init__(parent, fg_color="#1e2329", **kwargs)
    
    class GlassButton(ctk.CTkButton):
        pass
    
    class FloatingPanel(ctk.CTkFrame):
        def __init__(self, parent, *args, **kwargs):
            kwargs.pop('elevation', None)
            super().__init__(parent, fg_color="#1e2329", **kwargs)
    
    class GlassCard(ctk.CTkFrame):
        def __init__(self, parent, title="", value="", icon="", trend="", *args, **kwargs):
            super().__init__(parent, fg_color="#1e2329", **kwargs)
            ctk.CTkLabel(self, text=f"{icon} {title}: {value}").pack(pady=20)
    
    class GlassPanel(ctk.CTkFrame):
        def __init__(self, parent, title="", *args, **kwargs):
            super().__init__(parent, fg_color="#1e2329", **kwargs)
            self.content = ctk.CTkFrame(self, fg_color='transparent')
            self.content.pack(fill='both', expand=True)
    
    class AnimatedBarChart(ctk.CTkFrame):
        def __init__(self, *args, **kwargs): 
            super().__init__(*args, **kwargs)
        def set_data(self, *args): pass
    
    class AnimatedLineChart(ctk.CTkFrame):
        def __init__(self, *args, **kwargs): 
            super().__init__(*args, **kwargs)
        def set_data(self, *args): pass
    
    class ProgressBar(ctk.CTkFrame):
        def __init__(self, *args, **kwargs): 
            super().__init__(*args, **kwargs)
        def set_progress(self, *args): pass
        def winfo_exists(self): return True
    
    class DonutChart(ctk.CTkFrame):
        def __init__(self, *args, **kwargs): 
            super().__init__(*args, **kwargs)
        def set_data(self, *args): pass
    
    class SparklineChart(ctk.CTkFrame):
        def __init__(self, *args, **kwargs): 
            super().__init__(*args, **kwargs)
    
    class TypewriterLabel(ctk.CTkLabel):
        def __init__(self, parent, full_text="", *args, **kwargs):
            kwargs.pop('speed', None)
            super().__init__(parent, text=full_text, *args, **kwargs)
        def start_typing(self): pass
    
    class PulsingButton(ctk.CTkButton):
        def start_pulse(self): pass
    
    class RippleEffect:
        @staticmethod
        def create_ripple(*args): pass
    
    class WaveBackground:
        pass
    
    class GlowEffect:
        pass
    
    class AnimatedThemeTransition:
        pass
    
    class theme_manager:
        @staticmethod
        def get_current_theme():
            return {
                'colors': {
                    'bg_primary': '#0d1117',
                    'bg_secondary': '#161b22',
                    'bg_tertiary': '#1e2329',
                    'accent_primary': '#3b82f6',
                    'accent_secondary': '#00ff88',
                    'text_primary': '#f0f6fc',
                    'text_secondary': '#8b949e'
                }
            }
        
        @staticmethod
        def get_color(key):
            colors = {
                'bg_primary': '#0d1117',
                'bg_secondary': '#161b22',
                'bg_tertiary': '#1e2329',
                'accent_primary': '#3b82f6',
                'accent_secondary': '#00ff88',
                'text_primary': '#f0f6fc',
                'text_secondary': '#8b949e'
            }
            return colors.get(key, '#3b82f6')


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class AppConfig:
    """Configuration de l'application"""
    openai_api_key: str = ""
    theme: str = "dark"
    language: str = "fr"
    timeout: int = 30
    max_retries: int = 3
    debug_mode: bool = False

    def is_valid(self) -> bool:
        return len(self.openai_api_key) > 10


class ConfigManager:
    """Gestionnaire de configuration"""

    def __init__(self):
        self._config_path = Path("data/config.json")
        self.config = self._load_config()
        self.client = None
        self._tickers_db = {}
        self._load_tickers()
        self._setup_openai()

    def _load_config(self):
        """Charge la configuration"""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                app_settings = data.get("app_settings", {})
                api_settings = data.get("api_settings", {})
                
                return AppConfig(
                    openai_api_key=data.get("openai_api_key", ""),
                    theme=app_settings.get("theme", "dark"),
                    language=app_settings.get("language", "fr"),
                    timeout=api_settings.get("timeout", 30),
                    max_retries=api_settings.get("retries", 3),
                    debug_mode=app_settings.get("debug", False)
                )
            else:
                config = AppConfig()
                self._save_default_config(config)
                return config
                
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return AppConfig()
    
    def _save_default_config(self, config: AppConfig):
        """Sauvegarde la configuration par défaut"""
        try:
            self._config_path.parent.mkdir(exist_ok=True)
            
            data = {
                "openai_api_key": config.openai_api_key,
                "app_settings": {
                    "theme": config.theme,
                    "language": config.language,
                    "debug": config.debug_mode
                },
                "api_settings": {
                    "timeout": config.timeout,
                    "retries": config.max_retries
                }
            }
            
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")

    def _load_tickers(self):
        """Charge la base de tickers"""
        FALLBACK_MINIMAL = {
            "aapl": "AAPL", "apple": "AAPL",
            "msft": "MSFT", "microsoft": "MSFT",
            "googl": "GOOGL", "google": "GOOGL",
            "amzn": "AMZN", "amazon": "AMZN",
            "tsla": "TSLA", "tesla": "TSLA",
            "meta": "META", "facebook": "META",
        }

        try:
            if isinstance(EXT_INDEX, dict):
                normalized = {}
                for k, v in EXT_INDEX.items():
                    if isinstance(k, str) and isinstance(v, str):
                        key = k.strip().lower()
                        sym = v.strip().upper()
                        if key and sym:
                            normalized[key] = sym
                
                if normalized:
                    self._tickers_db = normalized
                    logger.info(f"{len(self._tickers_db)} tickers chargés")
                    return
            
            elif isinstance(EXT_INDEX, list):
                normalized = {}
                for item in EXT_INDEX:
                    if isinstance(item, dict):
                        sym = str(item.get("symbol", "")).strip().upper()
                        name = str(item.get("name", "")).strip().lower()
                        if sym:
                            if name:
                                normalized[name] = sym
                            normalized[sym.lower()] = sym
                
                if normalized:
                    self._tickers_db = normalized
                    logger.info(f"{len(self._tickers_db)} tickers chargés")
                    return

        except Exception as e:
            logger.error(f"Erreur chargement tickers: {e}")

        self._tickers_db = FALLBACK_MINIMAL
        logger.warning("Utilisation du fallback minimal de tickers")

    def _setup_openai(self):
        """Configure OpenAI"""
        try:
            if self.config.openai_api_key:
                self.client = openai.OpenAI(
                    api_key=self.config.openai_api_key, 
                    timeout=self.config.timeout
                )
                logger.info("OpenAI configuré avec succès")
            else:
                logger.warning("Clé OpenAI non configurée")
                self.client = None
        except Exception as e:
            logger.error(f"Erreur configuration OpenAI: {e}")
            self.client = None

    def is_openai_available(self) -> bool:
        return self.client is not None

    def get_client(self):
        return self.client

    def save_config(self, new_config: dict):
        """Sauvegarde la configuration"""
        try:
            data = {}
            if self._config_path.exists():
                try:
                    with open(self._config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            for key, value in new_config.items():
                if key.endswith("_api_key"):
                    data[key] = value
                elif key in ["theme", "language", "debug"]:
                    data.setdefault("app_settings", {})[key] = value
                elif key in ["timeout", "retries"]:
                    data.setdefault("api_settings", {})[key] = value

            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.config = self._load_config()
            self._setup_openai()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")


class DataManager:
    """Gestionnaire de données"""

    def __init__(self):
        self._data_dir = Path("data")
        self._ensure_data_directory()
        self._favorites = []
        self._tickers_db = {}
        
        self._load_favorites()
        self._load_tickers()

    def _ensure_data_directory(self):
        self._data_dir.mkdir(exist_ok=True)
        logger.debug("Dossier data vérifié/créé")

    def _load_favorites(self):
        fav_path = self._data_dir / "favoris.json"
        try:
            if fav_path.exists():
                with open(fav_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._favorites = [item for item in data 
                                    if isinstance(item, str) and item.strip()]
                    logger.info(f"{len(self._favorites)} favoris chargés")
                else:
                    self._favorites = []
            else:
                self._favorites = []
                self._save_favorites()
        except Exception as e:
            logger.error(f"Erreur chargement favoris: {e}")
            self._favorites = []

    def _save_favorites(self):
        try:
            fav_path = self._data_dir / "favoris.json"
            with open(fav_path, 'w', encoding='utf-8') as f:
                json.dump(self._favorites, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde favoris: {e}")

    def _load_tickers(self):
        """Charge les tickers depuis la base étendue ou fallback"""
        fallback_tickers = {
            "aapl": "AAPL", "apple": "AAPL",
            "msft": "MSFT", "microsoft": "MSFT",
            "googl": "GOOGL", "google": "GOOGL",
            "amzn": "AMZN", "amazon": "AMZN",
            "tsla": "TSLA", "tesla": "TSLA",
            "meta": "META", "facebook": "META"
        }

        try:
            self._tickers_db = SEARCH_INDEX
            logger.info(f"{len(self._tickers_db)} tickers chargés depuis la base étendue")
            return
        except (ImportError, NameError):
            pass

        try:
            json_path = self._data_dir / "tickers.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._tickers_db = {}
                    for item in data:
                        if isinstance(item, dict) and "symbol" in item:
                            symbol = str(item["symbol"]).strip().upper()
                            self._tickers_db[symbol.lower()] = symbol
                            if "name" in item:
                                name = str(item["name"]).strip().lower()
                                if name:
                                    self._tickers_db[name] = symbol
                    logger.info(f"{len(self._tickers_db)} tickers chargés depuis JSON")
                    return
        except Exception as e:
            logger.error(f"Erreur chargement tickers JSON: {e}")

        self._tickers_db = fallback_tickers
        logger.warning("Utilisation du fallback minimal de tickers")

    def add_favorite(self, ticker: str) -> bool:
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.strip().upper()
        if len(ticker) < 1 or len(ticker) > 10:
            return False
        
        if ticker not in self._favorites:
            self._favorites.append(ticker)
            self._save_favorites()
            logger.info(f"Favori ajouté: {ticker}")
            return True
        return False

    def remove_favorite(self, ticker: str) -> bool:
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.strip().upper()
        if ticker in self._favorites:
            self._favorites.remove(ticker)
            self._save_favorites()
            logger.info(f"Favori retiré: {ticker}")
            return True
        return False

    def get_favorites(self) -> List[str]:
        return self._favorites.copy()

    def find_ticker(self, query: str) -> Optional[str]:
        try:
            return find_ticker_func(query)
        except (ImportError, NameError):
            pass

        if not query:
            return None

        query = query.strip().lower()
        if query in self._tickers_db:
            return self._tickers_db[query]

        for key, ticker in self._tickers_db.items():
            if query in key:
                return ticker
        return None

    def get_ticker_suggestions(self, query: str, limit: int = 5) -> List[str]:
        try:
            return get_suggestions_func(query, limit)
        except (ImportError, NameError):
            pass

        if not query:
            return []

        query = query.strip().lower()
        suggestions = []

        for key, ticker in list(self._tickers_db.items())[:500]:
            if query in key and ticker not in suggestions:
                suggestions.append(ticker)
                if len(suggestions) >= limit:
                    break
        return suggestions


@lru_cache(maxsize=32)
def get_cached_ticker_data(ticker: str, cache_minute: int):
    """Cache les données pendant 1 minute"""
    try:
        return yf.Ticker(ticker).history(period="1d")
    except Exception as e:
        logger.error(f"Erreur récupération données {ticker}: {e}")
        return None


# === INSTANCES GLOBALES ===
config_manager = ConfigManager()
data_manager = DataManager()
client = config_manager.get_client()

# Configuration du thème avec support premium
if PREMIUM_EFFECTS_AVAILABLE:
    initial_theme = theme_manager.get_current_theme()
    COLORS = initial_theme['colors']
else:
    initial_theme_name = getattr(config_manager.config, 'theme', 'dark')
    THEMES = {
        "dark": {
            'bg_primary': '#0d1117',
            'bg_secondary': '#161b22',
            'bg_tertiary': '#1e2329',
            'bg_hover': '#2a3038',
            'accent_green': '#00ff88',
            'accent_red': '#ff3860',
            'accent_blue': '#3b82f6',
            'accent_yellow': '#ffaa00',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e',
            'border': '#30363d'
        },
        "light": {
            'bg_primary': '#f7f8fa',
            'bg_secondary': '#ffffff',
            'bg_tertiary': '#f1f3f5',
            'bg_hover': '#e9ecef',
            'accent_green': '#059669',
            'accent_red': '#ef4444',
            'accent_blue': '#2563eb',
            'accent_yellow': '#d97706',
            'text_primary': '#0b141a',
            'text_secondary': '#495057',
            'border': '#d0d7de'
        }
    }
    COLORS = dict(THEMES[initial_theme_name])

user_config = {
    "theme": getattr(config_manager.config, 'theme', 'dark'),
    "langue": "Français",
    "mode": "Long Terme",
    "alerte_active": True
}


# ============================================================================
# COMPOSANTS UI AVEC EFFETS PREMIUM
# ============================================================================
class PremiumAnalysisPanel(ctk.CTkFrame):
    """Panel d'analyse avec tous les effets visuels premium"""
    
    def __init__(self, parent):
        super().__init__(parent, fg_color='transparent')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Fond de particules animées si disponible
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.particle_bg = ParticleCanvas(self, theme='dark')
                self.particle_canvas = self.particle_bg.create()
            except Exception as e:
                logger.error(f"Erreur création particules: {e}")
        
        # Header avec effet glass
        self._create_header()
        
        # Conteneur principal scrollable
        self.scroll_container = ctk.CTkScrollableFrame(
            self,
            fg_color='transparent'
        )
        self.scroll_container.grid(row=1, column=0, sticky='nsew', padx=20, pady=20)
        
        # Zone de saisie du ticker
        self._create_input_section()
        
        # Zone d'affichage des résultats (initialement cachée)
        self.results_frame = None
    
    def _create_header(self):
        """Crée le header avec effet glassmorphism"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                header = GlassFrame(
                    self,
                    height=100,
                    border_glow=True
                )
            except:
                header = ctk.CTkFrame(self, fg_color=COLORS['bg_secondary'], height=100)
        else:
            header = ctk.CTkFrame(self, fg_color=COLORS['bg_secondary'], height=100)
        
        header.grid(row=0, column=0, sticky='ew', padx=20, pady=(20, 0))
        
        # Titre avec effet typewriter si disponible
        title_text = "Analyse FXI Premium - Intelligence Financière Avancée"
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.title_label = TypewriterLabel(
                    header,
                    full_text=title_text,
                    font=('Helvetica', 28, 'bold'),
                    text_color=COLORS.get('accent_primary', COLORS['accent_blue'])
                )
                self.title_label.pack(pady=30)
                self.title_label.start_typing()
            except:
                ctk.CTkLabel(
                    header, text=title_text,
                    font=('Helvetica', 24, 'bold'),
                    text_color=COLORS['accent_blue']
                ).pack(pady=30)
        else:
            ctk.CTkLabel(
                header, text=title_text,
                font=('Helvetica', 24, 'bold'),
                text_color=COLORS['accent_blue']
            ).pack(pady=30)
    
    def _create_input_section(self):
        """Crée la section de saisie avec effets"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                input_panel = FloatingPanel(
                    self.scroll_container,
                    elevation=5,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                input_panel = ctk.CTkFrame(
                    self.scroll_container, 
                    fg_color=COLORS['bg_secondary']
                )
        else:
            input_panel = ctk.CTkFrame(
                self.scroll_container, 
                fg_color=COLORS['bg_secondary']
            )
        
        input_panel.pack(pady=30, padx=50, fill='x')
        
        # Titre
        title = ctk.CTkLabel(
            input_panel,
            text="Entrez un symbole boursier",
            font=('Helvetica', 20, 'bold'),
            text_color=COLORS.get('text_primary', COLORS['text_primary'])
        )
        title.pack(pady=(30, 20))
        
        # Frame pour input et bouton
        input_frame = ctk.CTkFrame(input_panel, fg_color='transparent')
        input_frame.pack(pady=20)
        
        # Champ de saisie stylé
        self.ticker_entry = ctk.CTkEntry(
            input_frame,
            width=300,
            height=50,
            placeholder_text="Ex: AAPL, MSFT, GOOGL...",
            font=('Helvetica', 16),
            border_color=COLORS.get('accent_primary', COLORS['accent_blue']),
            border_width=2,
            corner_radius=12
        )
        self.ticker_entry.pack(side='left', padx=10)
        self.ticker_entry.bind('<Return>', lambda e: self._start_analysis())
        
        # Bouton d'analyse avec effet glow
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.analyze_btn = GlassButton(
                    input_frame,
                    text="Analyser",
                    width=200,
                    height=50,
                    font=('Helvetica', 16, 'bold'),
                    command=self._start_analysis
                )
            except:
                self.analyze_btn = ctk.CTkButton(
                    input_frame,
                    text="Analyser",
                    width=200,
                    height=50,
                    font=('Helvetica', 16, 'bold'),
                    command=self._start_analysis
                )
        else:
            self.analyze_btn = ctk.CTkButton(
                input_frame,
                text="Analyser",
                width=200,
                height=50,
                font=('Helvetica', 16, 'bold'),
                command=self._start_analysis
            )
        
        self.analyze_btn.pack(side='left', padx=10)
        
        # Effet ripple sur le bouton si disponible
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.analyze_btn.bind('<Button-1>', lambda e: RippleEffect.create_ripple(self.analyze_btn, e))
            except:
                pass
        
        # Sélecteur de mode
        mode_frame = ctk.CTkFrame(input_panel, fg_color='transparent')
        mode_frame.pack(pady=20)
        
        ctk.CTkLabel(
            mode_frame,
            text="Mode d'analyse:",
            font=('Helvetica', 14)
        ).pack(side='left', padx=10)
        
        self.mode_var = ctk.StringVar(value="Standard")
        mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            values=["Standard", "Conservative", "Aggressive"],
            variable=self.mode_var,
            width=200,
            height=40,
            font=('Helvetica', 14),
            fg_color=COLORS.get('accent_primary', COLORS['accent_blue']),
            button_color=COLORS.get('accent_secondary', COLORS['accent_green'])
        )
        mode_menu.pack(side='left', padx=10)
    
    def _start_analysis(self):
        """Démarre l'analyse avec animation"""
        ticker = self.ticker_entry.get().strip().upper()
        
        if not ticker:
            safe_show_notification("Veuillez entrer un symbole boursier", "warning")
            return
        
        # Désactiver le bouton et montrer loading
        self.analyze_btn.configure(state='disabled', text="Analyse en cours...")
        
        # Créer un indicateur de chargement
        self._show_loading()
        
        # Lancer l'analyse dans un thread pour ne pas bloquer l'UI
        import threading
        thread = threading.Thread(target=self._perform_analysis, args=(ticker,))
        thread.daemon = True
        thread.start()
    
    def _show_loading(self):
        """Affiche un indicateur de chargement animé"""
        if self.results_frame:
            self.results_frame.destroy()
        
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.loading_frame = FloatingPanel(
                    self.scroll_container,
                    elevation=3,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                self.loading_frame = ctk.CTkFrame(
                    self.scroll_container, 
                    fg_color=COLORS['bg_secondary']
                )
        else:
            self.loading_frame = ctk.CTkFrame(
                self.scroll_container, 
                fg_color=COLORS['bg_secondary']
            )
        
        self.loading_frame.pack(pady=30, padx=50, fill='both', expand=True)
        
        # Animation de chargement
        loading_text = "Analyse en cours... Collecte des données macro-économiques..."
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                loading_label = TypewriterLabel(
                    self.loading_frame,
                    full_text=loading_text,
                    font=('Helvetica', 16),
                    speed=30
                )
                loading_label.pack(pady=50)
                loading_label.start_typing()
            except:
                ctk.CTkLabel(
                    self.loading_frame, 
                    text=loading_text,
                    font=('Helvetica', 16)
                ).pack(pady=50)
        else:
            ctk.CTkLabel(
                self.loading_frame, 
                text=loading_text,
                font=('Helvetica', 16)
            ).pack(pady=50)
        
        # Barre de progression
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.progress = ProgressBar(self.loading_frame, width=500, height=30)
                self.progress.pack(pady=20)
                self._animate_progress()
            except:
                pass
    
    def _animate_progress(self, value: float = 0):
        """Anime la barre de progression"""
        if hasattr(self, 'progress') and self.progress.winfo_exists():
            try:
                self.progress.set_progress(value)
            except:
                pass
            if value < 95:
                self.after(100, lambda: self._animate_progress(value + 5))
    
    def _perform_analysis(self, ticker: str):
        """Effectue l'analyse (dans un thread)"""
        try:
           mode = self.mode_var.get()
           result = get_analysis(ticker, mode)
         
           # Mettre à jour l'UI dans le thread principal
           self.after(0, lambda: self._display_results(result, ticker, mode))
        
        except Exception as e:
            logger.error(f"Erreur analyse: {e}")
            self.after(0, lambda: self._show_error(f"Erreur d'analyse: {str(e)}"))  # LIGNE CORRIGÉE
      
    def _display_results(self, result: dict, ticker: str, mode: str):
        """Affiche les résultats avec tous les effets visuels"""
        # Supprimer le loading
        if hasattr(self, 'loading_frame'):
            self.loading_frame.destroy()
        
        # Réactiver le bouton
        self.analyze_btn.configure(state='normal', text="🚀 Analyser")
        
        # Créer le panel de résultats
        if self.results_frame:
            self.results_frame.destroy()
        
        self.results_frame = ctk.CTkFrame(
            self.scroll_container,
            fg_color='transparent'
        )
        self.results_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # 1. CARTES DE SYNTHÈSE (en haut)
        self._create_summary_cards(result, ticker)
        
        # 2. SCORE PRINCIPAL ANIMÉ
        self._create_main_score(result)
        
        # 3. GRAPHIQUE RADAR DES 5 DIMENSIONS
        self._create_radar_chart(result)
        
        # 4. JAUGES CIRCULAIRES PAR DIMENSION
        self._create_circular_gauges(result)
        
        # 5. GRAPHIQUE EN BARRES DES SCORES
        self._create_bar_chart(result)
        
        # 6. RAPPORT DÉTAILLÉ
        self._create_detailed_report(result, ticker, mode)
    
    def _create_summary_cards(self, result: dict, ticker: str):
        """Crée les cartes de synthèse en haut"""
        cards_container = ctk.CTkFrame(self.results_frame, fg_color='transparent')
        cards_container.pack(pady=20, fill='x')
        
        # Récupérer les données
        score = result.get('score_fxi', 50)
        recommandation = result.get('recommandation', 'N/A')
        confidence = result.get('confidence', 0)
        
        # Créer 4 cartes
        cards_data = [
            ("Score Global FXI", f"{score:.1f}", "📊"),
            ("Recommandation", recommandation, "🎯"),
            ("Confiance", f"{confidence:.0f}%", "✨"),
            ("Symbole", ticker, "💹")
        ]
        
        for i, (title, value, icon) in enumerate(cards_data):
            if PREMIUM_EFFECTS_AVAILABLE:
                try:
                    card = GlassCard(
                        cards_container,
                        title=title,
                        value=value,
                        icon=icon
                    )
                except:
                    card = ctk.CTkFrame(cards_container, fg_color=COLORS['bg_tertiary'])
                    ctk.CTkLabel(card, text=f"{icon} {title}: {value}").pack(pady=20)
            else:
                card = ctk.CTkFrame(cards_container, fg_color=COLORS['bg_tertiary'])
                ctk.CTkLabel(card, text=f"{icon} {title}: {value}", font=('Helvetica', 14, 'bold')).pack(pady=20)
            
            card.grid(row=0, column=i, padx=15, pady=10, sticky='ew')
        
        # Configuration de la grille
        for i in range(4):
            cards_container.grid_columnconfigure(i, weight=1)
    
    def _create_main_score(self, result: dict):
        """Crée le score principal animé"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                score_panel = FloatingPanel(
                    self.results_frame,
                    elevation=5,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                score_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        else:
            score_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        
        score_panel.pack(pady=30, fill='x')
        
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                score_widget = AnimatedScore(score_panel, fg_color='transparent')
                score_widget.pack(pady=40)
                score = result.get('score_fxi', 50)
                score_widget.animate_to(score)
            except Exception as e:
                logger.error(f"Erreur AnimatedScore: {e}")
                self._create_static_score(score_panel, result)
        else:
            self._create_static_score(score_panel, result)
    
    def _create_static_score(self, parent, result: dict):
        """Crée un score statique si les effets ne sont pas disponibles"""
        score = result.get('score_fxi', 50)
        ctk.CTkLabel(
            parent,
            text=f"{score:.1f}",
            font=('Helvetica', 48, 'bold'),
            text_color=COLORS['accent_green'] if score >= 70 else COLORS['accent_blue']
        ).pack(pady=40)
    
    def _create_radar_chart(self, result: dict):
        """Crée le graphique radar des 5 dimensions"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                radar_panel = FloatingPanel(
                    self.results_frame,
                    elevation=5,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                radar_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        else:
            radar_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        
        radar_panel.pack(pady=30, fill='both', expand=True)
        
        title = ctk.CTkLabel(
            radar_panel,
            text="📈 Analyse Multi-Dimensionnelle",
            font=('Helvetica', 20, 'bold')
        )
        title.pack(pady=20)
        
        # Radar chart
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                scores_detailles = result.get('scores_detailles', {})
                values = [
                    scores_detailles.get('technique', 50),
                    scores_detailles.get('fondamental', 50),
                    scores_detailles.get('sentiment', 50),
                    scores_detailles.get('risque', 50),
                    scores_detailles.get('macro', 50)
                ]
                
                radar = RadarChart(radar_panel, size=400, dimensions=5)
                radar.pack(pady=20)
                radar.set_values(values, animated=True)
            except Exception as e:
                logger.error(f"Erreur RadarChart: {e}")
                self._create_scores_list(radar_panel, result)
        else:
            self._create_scores_list(radar_panel, result)
    
    def _create_scores_list(self, parent, result: dict):
        """Crée une liste de scores si le radar n'est pas disponible"""
        scores_detailles = result.get('scores_detailles', {})
        labels = ["Technique", "Fondamental", "Sentiment", "Risque", "Macro"]
        
        for label in labels:
            score = scores_detailles.get(label.lower(), 50)
            ctk.CTkLabel(
                parent,
                text=f"{label}: {score:.1f}/100",
                font=('Helvetica', 14)
            ).pack(pady=5)
    
    def _create_circular_gauges(self, result: dict):
        """Crée les jauges circulaires pour chaque dimension"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                gauges_panel = FloatingPanel(
                    self.results_frame,
                    elevation=5,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                gauges_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        else:
            gauges_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        
        gauges_panel.pack(pady=30, fill='x')
        
        title = ctk.CTkLabel(
            gauges_panel,
            text="🎯 Scores Détaillés par Dimension",
            font=('Helvetica', 20, 'bold')
        )
        title.pack(pady=20)
        
        gauges_frame = ctk.CTkFrame(gauges_panel, fg_color='transparent')
        gauges_frame.pack(pady=20)
        
        scores_detailles = result.get('scores_detailles', {})
        dimensions = [
            ("Technique", scores_detailles.get('technique', 50), "🔧"),
            ("Fondamental", scores_detailles.get('fondamental', 50), "💰"),
            ("Sentiment", scores_detailles.get('sentiment', 50), "🎭"),
            ("Risque", scores_detailles.get('risque', 50), "⚡"),
            ("Macro-Éco", scores_detailles.get('macro', 50), "🌍")
        ]
        
        for i, (label, value, icon) in enumerate(dimensions):
            container = ctk.CTkFrame(gauges_frame, fg_color='transparent')
            container.grid(row=0, column=i, padx=20, pady=10)
            
            # Icône
            icon_label = ctk.CTkLabel(
                container,
                text=icon,
                font=('Helvetica', 32)
            )
            icon_label.pack()
            
            # Jauge ou score simple
            if PREMIUM_EFFECTS_AVAILABLE:
                try:
                    gauge = CircularGauge(container, size=140)
                    gauge.pack(pady=10)
                    gauge.set_value(value, animated=True)
                except:
                    ctk.CTkLabel(
                        container,
                        text=f"{value:.1f}",
                        font=('Helvetica', 24, 'bold')
                    ).pack(pady=10)
            else:
                ctk.CTkLabel(
                    container,
                    text=f"{value:.1f}",
                    font=('Helvetica', 24, 'bold')
                ).pack(pady=10)
            
            # Label
            label_widget = ctk.CTkLabel(
                container,
                text=label,
                font=('Helvetica', 14, 'bold')
            )
            label_widget.pack(pady=5)
    
    def _create_bar_chart(self, result: dict):
        """Crée le graphique en barres des scores"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                chart_panel = FloatingPanel(
                    self.results_frame,
                    elevation=5,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                chart_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        else:
            chart_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        
        chart_panel.pack(pady=30, fill='both', expand=True)
        
        title = ctk.CTkLabel(
            chart_panel,
            text="📊 Comparaison des Scores",
            font=('Helvetica', 20, 'bold')
        )
        title.pack(pady=20)
        
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                scores_detailles = result.get('scores_detailles', {})
                labels = ["Technique", "Fondamental", "Sentiment", "Risque", "Macro"]
                values = [
                    scores_detailles.get('technique', 50),
                    scores_detailles.get('fondamental', 50),
                    scores_detailles.get('sentiment', 50),
                    scores_detailles.get('risque', 50),
                    scores_detailles.get('macro', 50)
                ]
                colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
                
                chart = AnimatedBarChart(chart_panel, width=800, height=400)
                chart.pack(pady=20)
                chart.set_data(labels, values, colors)
            except Exception as e:
                logger.error(f"Erreur AnimatedBarChart: {e}")
                self._create_scores_list(chart_panel, result)
        else:
            self._create_scores_list(chart_panel, result)
    
    def _create_detailed_report(self, result: dict, ticker: str, mode: str):
        """Crée le rapport détaillé"""
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                report_panel = FloatingPanel(
                    self.results_frame,
                    elevation=5,
                    fg_color=COLORS.get('bg_secondary', COLORS['bg_secondary'])
                )
            except:
                report_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        else:
            report_panel = ctk.CTkFrame(self.results_frame, fg_color=COLORS['bg_secondary'])
        
        report_panel.pack(pady=30, fill='both', expand=True)
        
        title = ctk.CTkLabel(
            report_panel,
            text="📄 Rapport Détaillé",
            font=('Helvetica', 20, 'bold')
        )
        title.pack(pady=20)
        
        # Générer le rapport
        try:
            rapport_texte = generer_rapport_v3(ticker, mode, result)
        except:
            rapport_texte = "Rapport non disponible"
        
        # Zone de texte scrollable
        report_text = ctk.CTkTextbox(
            report_panel,
            width=900,
            height=500,
            font=('Courier', 12),
            fg_color=COLORS.get('bg_tertiary', COLORS['bg_tertiary']),
            text_color=COLORS.get('text_primary', COLORS['text_primary']),
            wrap='word'
        )
        report_text.pack(pady=20, padx=20)
        report_text.insert('1.0', rapport_texte)
        report_text.configure(state='disabled')
        
        # Boutons d'action
        buttons_frame = ctk.CTkFrame(report_panel, fg_color='transparent')
        buttons_frame.pack(pady=20)
        
        if PREMIUM_EFFECTS_AVAILABLE:
            try:
                export_btn = GlassButton(
                    buttons_frame,
                    text="💾 Exporter PDF",
                    width=200,
                    command=lambda: self._export_report(rapport_texte, ticker)
                )
            except:
                export_btn = ctk.CTkButton(
                    buttons_frame,
                    text="💾 Exporter PDF",
                    width=200,
                    command=lambda: self._export_report(rapport_texte, ticker)
                )
        else:
            export_btn = ctk.CTkButton(
                buttons_frame,
                text="💾 Exporter PDF",
                width=200,
                command=lambda: self._export_report(rapport_texte, ticker)
            )
        
        export_btn.pack(side='left', padx=10)
    
    def _show_error(self, message: str):
        """Affiche un message d'erreur stylé"""
        if hasattr(self, 'loading_frame'):
            self.loading_frame.destroy()
        
        self.analyze_btn.configure(state='normal', text="🚀 Analyser")
        
        error_panel = ctk.CTkFrame(
            self.scroll_container,
            fg_color='#ff444433'
        )
        error_panel.pack(pady=30, padx=50)
        
        error_label = ctk.CTkLabel(
            error_panel,
            text=f"❌ {message}",
            font=('Helvetica', 16, 'bold'),
            text_color='#ff4444'
        )
        error_label.pack(pady=40, padx=40)
        
        # Auto-suppression après 3 secondes
        self.after(3000, error_panel.destroy)
    
    def _export_report(self, rapport: str, ticker: str):
        """Exporte le rapport"""
        safe_exporter_rapport()
    
    def destroy(self):
        """Nettoyage lors de la destruction"""
        if hasattr(self, 'particle_bg') and PREMIUM_EFFECTS_AVAILABLE:
            try:
                self.particle_bg.stop()
            except:
                pass
        super().destroy()


class SafeMetricCard(ctk.CTkFrame):
    """Carte métrique avec gestion d'erreurs"""

    def __init__(self, parent, title: str, value, change: float, icon: str, color: str = "#00ff88"):
        super().__init__(parent, fg_color=COLORS['bg_tertiary'], corner_radius=15, height=120)
        self._is_destroyed = False

        try:
            if not isinstance(title, str):
                title = str(title)
            if not isinstance(change, (int, float)):
                change = 0.0
            self._setup_ui(title, value, change, icon, color)
        except Exception as e:
            logger.error(f"Erreur création MetricCard: {e}")
            self._create_error_display()

    def _setup_ui(self, title: str, value, change: float, icon: str, color: str):
        icon_label = ctk.CTkLabel(self, text=icon, font=("Segoe UI", 24))
        icon_label.place(x=20, y=20)

        title_label = ctk.CTkLabel(
            self, text=title, font=("Segoe UI", 12),
            text_color=COLORS['text_secondary']
        )
        title_label.place(x=60, y=25)

        self.value_label = ctk.CTkLabel(
            self, text="0", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        self.value_label.place(x=20, y=55)

        change_color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
        change_text = f"↗ {change:.1f}%" if change >= 0 else f"↘ {abs(change):.1f}%"

        change_label = ctk.CTkLabel(
            self, text=change_text, font=("Segoe UI", 11),
            text_color=change_color
        )
        change_label.place(x=20, y=90)

        self._animate_counter_safe(0, value)

    def _create_error_display(self):
        error_label = ctk.CTkLabel(
            self, text="Erreur d'affichage", font=("Segoe UI", 12),
            text_color=COLORS['accent_red']
        )
        error_label.place(relx=0.5, rely=0.5, anchor="center")

    def _animate_counter_safe(self, start: int, end):
        def animate():
            try:
                end_val = int(end) if isinstance(end, (int, float)) else 0
                if end_val <= 0:
                    self._update_value_safe(str(end_val))
                    return

                steps = max(1, min(20, end_val // 10))
                increment = max(1, end_val // steps)
                current = start

                while current < end_val and not self._is_destroyed:
                    if self.winfo_exists():
                        self.after_idle(lambda v=current: self._update_value_safe(f"{v:,}"))
                        current += increment
                        time.sleep(0.03)
                    else:
                        break

                if not self._is_destroyed and self.winfo_exists():
                    self.after_idle(lambda: self._update_value_safe(f"{end_val:,}"))
            except Exception as e:
                logger.error(f"Erreur animation MetricCard: {e}")

        threading.Thread(target=animate, daemon=True).start()

    def _update_value_safe(self, text: str):
        try:
            if self.winfo_exists() and hasattr(self, 'value_label'):
                self.value_label.configure(text=text)
        except:
            pass

    def destroy(self):
        self._is_destroyed = True
        try:
            super().destroy()
        except:
            pass


class ModernSidebar(ctk.CTkFrame):
    """Sidebar moderne avec navigation et menu hamburger"""

    def __init__(self, parent, main_frame):
        super().__init__(parent, width=250, fg_color=COLORS['bg_secondary'])
        self.main_frame = main_frame
        self.pack_propagate(False)
        self.buttons = []
        self.is_collapsed = False
        self.sidebar_width = 250
        self.collapsed_width = 0
        self.animation_steps = 1  # 1 step = instantané, pas d'animation
        self.animation_speed = 1  # Minimal delay

        try:
            self.setup_header()
            self.setup_navigation()
            self.select_default_button()
        except Exception as e:
            logger.error(f"Erreur création Sidebar: {e}")
            self._create_error_sidebar()

    def setup_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(pady=20)

        header = ctk.CTkLabel(
            header_frame, text="HelixOne", font=("Segoe UI", 24, "bold"),
            text_color=COLORS['accent_blue']
        )
        header.pack()

        subtitle = ctk.CTkLabel(
            header_frame, text="Trading Intelligence", font=("Segoe UI", 10),
            text_color=COLORS['text_secondary']
        )
        subtitle.pack()

        separator = ctk.CTkFrame(self, height=1, fg_color=COLORS['border'])
        separator.pack(fill="x", padx=20, pady=10)

    def setup_navigation(self):
        menu_items = [
            ("🏠", "Dashboard", self.show_dashboard),
            ("🔍", "Recherche", self.show_search),
            ("📊", "Mon Portfolio", self.show_portfolio),
            ("✨", "Analyse Premium", self.show_premium_analysis),
            ("📈", "Graphiques", self.show_charts),
            ("🎲", "Scénarios", self.show_scenarios),
            ("🎯", "Alertes", self.show_alerts),
            ("📚", "Formation", self.show_formation),
            ("⚙️", "Paramètres", self.show_settings),
            ("💬", "Communauté", self.show_community)
        ]

        for icon, text, command in menu_items:
            try:
                btn = self.create_menu_button(icon, text, command)
                btn.pack(fill="x", padx=15, pady=3)
                self.buttons.append(btn)
            except Exception as e:
                logger.error(f"Erreur création bouton {text}: {e}")

    def create_menu_button(self, icon: str, text: str, command):
        def safe_command():
            try:
                command()
            except Exception as e:
                logger.error(f"Erreur exécution commande {text}: {e}")
                self.show_error_message(f"Erreur dans {text}")

        btn = ctk.CTkButton(
            self, text=f"{icon}  {text}", font=("Segoe UI", 14),
            height=45, fg_color="transparent", 
            text_color=COLORS['text_secondary'],
            hover_color=COLORS['bg_hover'], anchor="w",
            command=safe_command
        )
        return btn

    def _create_error_sidebar(self):
        error_label = ctk.CTkLabel(
            self, text="Erreur navigation", font=("Segoe UI", 14),
            text_color=COLORS['accent_red']
        )
        error_label.pack(pady=50)

    def select_default_button(self):
        if self.buttons:
            try:
                self.buttons[0].configure(
                    fg_color=COLORS['bg_hover'],
                    text_color=COLORS['accent_green']
                )
            except:
                pass

    def reset_buttons(self):
        for btn in self.buttons:
            try:
                btn.configure(
                    fg_color="transparent",
                    text_color=COLORS['text_secondary']
                )
            except:
                continue

    def toggle_sidebar(self):
        """Toggle la sidebar avec animation slide"""
        if self.is_collapsed:
            self.expand_sidebar()
        else:
            self.collapse_sidebar()

    def collapse_sidebar(self):
        """Masque la sidebar avec animation"""
        self.is_collapsed = True
        self.animate_sidebar(self.sidebar_width, self.collapsed_width)

    def expand_sidebar(self):
        """Affiche la sidebar avec animation"""
        self.is_collapsed = False
        self.animate_sidebar(self.collapsed_width, self.sidebar_width)

    def animate_sidebar(self, start_width, end_width):
        """Anime la sidebar de start_width vers end_width"""
        step_size = (end_width - start_width) / self.animation_steps
        current_step = 0

        def animate_step():
            nonlocal current_step
            if current_step < self.animation_steps:
                current_step += 1
                new_width = int(start_width + (step_size * current_step))
                self.configure(width=new_width)
                self.after(self.animation_speed, animate_step)
            else:
                # Animation terminée
                self.configure(width=end_width)
                if self.is_collapsed:
                    # Masquer complètement
                    self.pack_forget()
                logger.info(f"✅ Sidebar {'masquée' if self.is_collapsed else 'affichée'}")

        # Si on expand, afficher d'abord le frame
        if not self.is_collapsed and not self.winfo_viewable():
            self.pack(side="left", fill="y", before=self.main_frame)

        animate_step()

    def show_dashboard(self):
        self.reset_buttons()
        if self.buttons:
            self.buttons[0].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_dashboard()

    def show_search(self):
        self.reset_buttons()
        if len(self.buttons) > 1:
            self.buttons[1].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_recherche()

    def show_portfolio(self):
        """Affiche le panel d'analyse de portfolio avec ML"""
        self.reset_buttons()
        if len(self.buttons) > 2:
            self.buttons[2].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_portfolio()

    def show_premium_analysis(self):
        """Affiche le panel d'analyse premium"""
        self.reset_buttons()
        if len(self.buttons) > 3:
            self.buttons[3].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )

        try:
            safe_clear_main_frame()
            if PREMIUM_EFFECTS_AVAILABLE:
                premium_panel = PremiumAnalysisPanel(self.main_frame)
                premium_panel.pack(fill='both', expand=True)
                logger.info("✨ Panel Premium affiché")
            else:
                # Fallback vers l'analyse standard
                safe_afficher_recherche()
                safe_show_notification("Effets premium non disponibles, utilisation de l'interface standard", "info")
        except Exception as e:
            logger.error(f"Erreur affichage panel premium: {e}")
            self.show_error_message("Erreur d'affichage du panel premium")

    def show_charts(self):
        self.reset_buttons()
        if len(self.buttons) > 4:
            self.buttons[4].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_graphiques()

    def show_scenarios(self):
        """Afficher le panel de simulation de scénarios"""
        self.reset_buttons()
        if len(self.buttons) > 5:
            self.buttons[5].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )

        try:
            safe_clear_main_frame()

            # Importer et afficher le ScenarioPanel
            from src.interface.scenario_panel import ScenarioPanel

            # Récupérer le client API
            try:
                from helixone_client import HelixOneClient
                client = HelixOneClient()
                # Configurer le token
                from auth_session import get_auth_token
                token = get_auth_token()
                if token:
                    client.token = token
            except Exception as e:
                logger.warning(f"Impossible de configurer le client API: {e}")
                client = None

            # Créer et afficher le panel
            scenario_panel = ScenarioPanel(
                self.main_frame,
                helixone_client=client
            )
            scenario_panel.pack(fill="both", expand=True)

            logger.info("✅ Panel Scénarios affiché")

        except Exception as e:
            logger.error(f"Erreur affichage Scénarios: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.show_error_message("Erreur lors de l'affichage des scénarios")

    def show_alerts(self):
        self.reset_buttons()
        if len(self.buttons) > 6:
            self.buttons[6].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_zone_alerte()

    def show_formation(self):
        self.reset_buttons()
        if len(self.buttons) > 7:
            self.buttons[7].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )

        try:
            safe_clear_main_frame()
            afficher_formation_commerciale(self.main_frame)
        except ImportError as e:
            logger.error(f"Module formation non trouvé: {e}")
            self.show_error_message("Module formation non disponible")
        except Exception as e:
            logger.error(f"Erreur chargement formation: {e}")
            self.show_error_message("Erreur de chargement de la formation")

    def show_settings(self):
        self.reset_buttons()
        if len(self.buttons) > 8:
            self.buttons[8].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_parametres()

    def show_community(self):
        self.reset_buttons()
        if len(self.buttons) > 9:
            self.buttons[9].configure(
                fg_color=COLORS['bg_hover'],
                text_color=COLORS['accent_green']
            )
        safe_afficher_communaute()

    def show_error_message(self, message: str):
        try:
            safe_clear_main_frame()

            error_frame = ctk.CTkFrame(self.main_frame, fg_color=COLORS['bg_secondary'])
            error_frame.pack(fill="both", expand=True, padx=20, pady=20)

            ctk.CTkLabel(
                error_frame, text=f"⚠️ {message}", font=("Segoe UI", 16),
                text_color=COLORS['accent_red']
            ).pack(pady=50)

            ctk.CTkButton(
                error_frame, text="Retour au Dashboard",
                command=self.show_dashboard, fg_color=COLORS['accent_blue']
            ).pack(pady=20)

        except Exception as e:
            logger.error(f"Erreur affichage message d'erreur: {e}")



# === NOTIFICATIONS ===
class SafeToastNotification(ctk.CTkToplevel):
    """Notifications toast sécurisées"""

    def __init__(self, parent, message: str, type: str = "info", duration: int = 3000):
        try:
            super().__init__(parent)
            self.overrideredirect(True)
            self.configure(fg_color=COLORS['bg_tertiary'])
            self.attributes('-topmost', True)
            self._setup_position()
            self._setup_content(message, type)
            self._schedule_close(duration)
            self._animate_slide_in()
        except Exception as e:
            logger.error(f"Erreur création notification: {e}")
            try:
                self.destroy()
            except:
                pass

    def _setup_position(self):
        try:
            self.update_idletasks()
            x = self.winfo_screenwidth() - 350
            y = self.winfo_screenheight() - 150
            self.geometry(f"320x80+{x}+{y}")
        except:
            self.geometry("320x80+100+100")

    def _setup_content(self, message: str, type: str):
        colors = {
            "success": COLORS['accent_green'],
            "error": COLORS['accent_red'],
            "warning": COLORS['accent_yellow'],
            "info": COLORS['accent_blue']
        }
        icons = {"success": "✓", "error": "✗", "warning": "⚠", "info": "ℹ"}

        frame = ctk.CTkFrame(self, fg_color=COLORS['bg_hover'], corner_radius=10)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        icon_label = ctk.CTkLabel(
            frame, text=icons.get(type, "ℹ"), font=("Segoe UI", 20),
            text_color=colors.get(type, COLORS['accent_blue'])
        )
        icon_label.place(x=15, y=25)

        display_message = message[:100] + "..." if len(message) > 100 else message
        msg_label = ctk.CTkLabel(
            frame, text=display_message, font=("Segoe UI", 12),
            text_color=COLORS['text_primary']
        )
        msg_label.place(x=50, y=28)

    def _schedule_close(self, duration: int):
        try:
            self.after(duration, self._safe_destroy)
        except:
            pass

    def _animate_slide_in(self):
        try:
            self.attributes('-alpha', 0)
            for i in range(1, 11):
                self.after(i * 30, lambda a=i/10: self._set_alpha_safe(a))
        except:
            pass

    def _set_alpha_safe(self, alpha: float):
        try:
            if self.winfo_exists():
                self.attributes('-alpha', alpha)
        except:
            pass

    def _safe_destroy(self):
        try:
            if self.winfo_exists():
                self.destroy()
        except:
            pass


# ============================================================================
# VARIABLES GLOBALES CONTROLEES
# ============================================================================
class AppState:
    def __init__(self):
        self.main_frame = None
        self.sidebar = None
        self.text_box = None
        self.entry = None
        self.suggestion_frame = None
        self.notebook = None
        self.tab_graphiques = None
        self.langue_var = None
        self.mode_var = None
        self.settings_vars = {}
        self.app = None
        self.container = None


app_state = AppState()

# Compatibilité avec l'ancien code
main_frame = None
sidebar = None
text_box = None
auth_manager_global = None  # Pour stocker auth_manager et récupérer le token
entry = None
suggestion_frame = None
notebook = None
tab_graphiques = None
langue_var = None
mode_var = None

# Cartes métriques pour IBKR
metric_cards = {
    'portfolio': None,
    'pnl': None,
    'trades': None,
    'winrate': None
}

# URL du microservice IBKR
IBKR_MICROSERVICE_URL = "http://127.0.0.1:8001"


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def safe_clear_main_frame():
    """Vide le main frame de manière sécurisée"""
    try:
        global main_frame
        if main_frame and main_frame.winfo_exists():
            for widget in main_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    continue
    except Exception as e:
        logger.error(f"Erreur nettoyage main_frame: {e}")


def safe_show_notification(message: str, type: str = "info"):
    """Affiche une notification de manière sécurisée"""
    try:
        parent = None
        if app_state.app and app_state.app.winfo_exists():
            parent = app_state.app

        if parent:
            SafeToastNotification(parent, message, type)
        else:
            logger.info(f"NOTIFICATION [{type.upper()}]: {message}")
    except Exception as e:
        logger.error(f"Erreur affichage notification: {e}")
        print(f"NOTIFICATION [{type.upper()}]: {message}")


def _apply_runtime_theme(theme_name: str):
    """Applique la palette et rafraîchit l'UI"""
    try:
        theme_name = theme_name.lower()
        if theme_name not in THEMES:
            theme_name = "dark"

        ctk.set_appearance_mode("light" if theme_name == "light" else "dark")
        COLORS.clear()
        COLORS.update(THEMES[theme_name])

        if app_state.main_frame and app_state.main_frame.winfo_exists():
            app_state.main_frame.configure(fg_color=COLORS['bg_primary'])
            safe_clear_main_frame()
            safe_afficher_dashboard()

        if app_state.sidebar and app_state.sidebar.winfo_exists():
            try:
                app_state.sidebar.configure(fg_color=COLORS['bg_secondary'])
                for btn in getattr(app_state.sidebar, "buttons", []):
                    try:
                        btn.configure(
                            fg_color="transparent",
                            text_color=COLORS['text_secondary'],
                            hover_color=COLORS['bg_hover']
                        )
                    except:
                        pass
            except:
                pass
    except Exception as e:
        logger.error(f"Erreur application thème: {e}")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def launch_main_app_with_auth(app, auth_manager, logout_callback):
    """Lance l'application principale avec authentification"""
    global main_frame, sidebar, langue_var, mode_var, auth_manager_global

    try:
        # Sauvegarder auth_manager globalement pour l'utiliser dans les analyses ML
        auth_manager_global = auth_manager
        logger.info("Démarrage de l'application HelixOne avec authentification")

        # === MODE DEV : Définir le token de test ===
        if os.environ.get("HELIXONE_DEV") == "1":
            from auth_session import set_auth_token
            dev_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjI2ZjI0MDctNGY2Yi00ODMyLWJjMTQtZGZhNzQ4M2JmY2Y0IiwiZW1haWwiOiJ0ZXN0QGhlbGl4b25lLmNvbSIsImV4cCI6MTc5MTkzMDA2N30.DDnZTWxmHCfPW6mVJrhKCU0HJeD7vCxcPTTIXwjmq5M"
            set_auth_token(dev_token)
            logger.info("✅ MODE DEV: Token d'authentification défini")
        else:
            # Mode production : utiliser le token de l'auth_manager
            from auth_session import set_auth_token
            if hasattr(auth_manager, 'token') and auth_manager.token:
                set_auth_token(auth_manager.token)
                logger.info("✅ Token d'authentification récupéré depuis auth_manager")

        # Récupérer les infos utilisateur
        user = auth_manager.get_current_user()
        license_info = auth_manager.get_license_info()
        
        logger.info(f"Utilisateur connecté: {user['email']}")
        logger.info(f"Licence: {license_info['license_type']} - {license_info['days_remaining']} jours")

        # Application du thème
        initial_theme = getattr(config_manager.config, 'theme', 'dark')
        initial_theme = "light" if initial_theme.lower() == "light" else "dark"
        COLORS.clear()
        COLORS.update(THEMES[initial_theme])
        ctk.set_appearance_mode(initial_theme)

        # Container principal
        container = ctk.CTkFrame(app, fg_color=COLORS['bg_primary'])
        container.pack(fill="both", expand=True)

        # Main frame
        main_frame = ctk.CTkFrame(container, fg_color=COLORS['bg_primary'])
        app_state.main_frame = main_frame
        app_state.app = app
        app_state.container = container

        # Sidebar avec bouton de déconnexion
        try:
            sidebar = ModernSidebar(container, main_frame)
            sidebar.pack(side="left", fill="y")
            app_state.sidebar = sidebar

            # Créer le bouton hamburger qui reste visible
            hamburger_frame = ctk.CTkFrame(
                container,
                width=50,
                height=50,
                fg_color=COLORS['bg_secondary'],
                corner_radius=10
            )
            hamburger_frame.place(x=10, y=10)

            hamburger_btn = ctk.CTkButton(
                hamburger_frame,
                text="☰",
                width=40,
                height=40,
                font=("Segoe UI", 24),
                fg_color=COLORS['accent_blue'],
                hover_color=COLORS['accent_green'],
                command=sidebar.toggle_sidebar,
                corner_radius=8
            )
            hamburger_btn.pack(padx=5, pady=5)

            # Ajouter le bouton de déconnexion en bas de la sidebar
            separator = ctk.CTkFrame(sidebar, height=1, fg_color=COLORS['border'])
            separator.pack(fill="x", padx=20, pady=20, side="bottom")
            
            # Afficher les infos utilisateur
            user_info_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            user_info_frame.pack(side="bottom", fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(
                user_info_frame,
                text=f"👤 {user['first_name']} {user['last_name']}",
                font=("Segoe UI", 11, "bold"),
                text_color=COLORS['text_primary']
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                user_info_frame,
                text=f"📧 {user['email']}",
                font=("Segoe UI", 9),
                text_color=COLORS['text_secondary']
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                user_info_frame,
                text=f"📜 {license_info['license_type'].upper()} - {license_info['days_remaining']} jours",
                font=("Segoe UI", 9),
                text_color=COLORS['accent_green']
            ).pack(anchor="w", pady=(5, 0))
            
            # Bouton de déconnexion
            logout_btn = ctk.CTkButton(
                sidebar,
                text="🚪 Déconnexion",
                font=("Segoe UI", 13, "bold"),
                height=45,
                fg_color=COLORS['accent_red'],
                hover_color="#d32f2f",
                command=logout_callback
            )
            logout_btn.pack(side="bottom", fill="x", padx=15, pady=10)
            
            logger.info("Sidebar créée avec succès")
        except Exception as e:
            logger.error(f"Erreur création sidebar: {e}")
            sidebar = create_fallback_sidebar(container)
            sidebar.pack(side="left", fill="y")

        # Main frame pack
        main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Variables de configuration
        langue_var = ctk.StringVar(value=user_config["langue"])
        mode_var = ctk.StringVar(value=user_config["mode"])
        app_state.langue_var = langue_var
        app_state.mode_var = mode_var

        # Raccourcis clavier
        app.bind('<Control-q>', lambda e: logout_callback())
        app.bind('<Control-n>', lambda e: safe_afficher_recherche())
        app.bind('<F5>', lambda e: safe_afficher_indices_boursiers())
        app.bind('<Escape>', lambda e: safe_clear_main_frame())

        # Bouton chat
        try:
            if config_manager.is_openai_available():
                chat_btn = ctk.CTkButton(
                    app, text="💬", width=50, height=50, corner_radius=25,
                    font=("Segoe UI", 20), fg_color=COLORS['accent_blue'],
                    hover_color=COLORS['accent_green'],
                    command=lambda: safe_create_assistant_chat(app)
                )
                chat_btn.place(relx=0.97, rely=0.97, anchor="se")
                logger.info("Bouton chat créé")
        except Exception as e:
            logger.error(f"Erreur création bouton chat: {e}")

        # Afficher le dashboard
        safe_afficher_dashboard()
        logger.info("Application HelixOne démarrée avec succès")

    except Exception as e:
        logger.error(f"Erreur critique au démarrage: {e}")
        create_emergency_ui(app)


# Maintenir la compatibilité avec l'ancienne fonction
def launch_main_app(app, auth_manager=None, logout_callback=None):
    """Fonction d'entrée compatible avec et sans authentification"""
    if auth_manager and logout_callback:
        # Mode avec authentification
        return launch_main_app_with_auth(app, auth_manager, logout_callback)
    else:
        # Mode sans authentification (ancien comportement)
        return launch_main_app_old(app)

def launch_main_app_old(app):
    """Ancienne fonction pour compatibilité"""
    global main_frame, sidebar, langue_var, mode_var

    try:
        logger.info("Démarrage de l'application HelixOne")

        initial_theme = getattr(config_manager.config, 'theme', 'dark')
        initial_theme = "light" if initial_theme.lower() == "light" else "dark"
        COLORS.clear()
        COLORS.update(THEMES[initial_theme])
        ctk.set_appearance_mode(initial_theme)

        container = ctk.CTkFrame(app, fg_color=COLORS['bg_primary'])
        container.pack(fill="both", expand=True)

        main_frame = ctk.CTkFrame(container, fg_color=COLORS['bg_primary'])
        app_state.main_frame = main_frame
        app_state.app = app
        app_state.container = container

        try:
            sidebar = ModernSidebar(container, main_frame)
            sidebar.pack(side="left", fill="y")
            app_state.sidebar = sidebar

            # Créer le bouton hamburger qui reste visible
            hamburger_frame = ctk.CTkFrame(
                container,
                width=50,
                height=50,
                fg_color=COLORS['bg_secondary'],
                corner_radius=10
            )
            hamburger_frame.place(x=10, y=10)

            hamburger_btn = ctk.CTkButton(
                hamburger_frame,
                text="☰",
                width=40,
                height=40,
                font=("Segoe UI", 24),
                fg_color=COLORS['accent_blue'],
                hover_color=COLORS['accent_green'],
                command=sidebar.toggle_sidebar,
                corner_radius=8
            )
            hamburger_btn.pack(padx=5, pady=5)

            logger.info("Sidebar créée avec succès")
        except Exception as e:
            logger.error(f"Erreur création sidebar: {e}")
            sidebar = create_fallback_sidebar(container)
            sidebar.pack(side="left", fill="y")

        main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        langue_var = ctk.StringVar(value=user_config["langue"])
        mode_var = ctk.StringVar(value=user_config["mode"])
        app_state.langue_var = langue_var
        app_state.mode_var = mode_var

        app.bind('<Control-q>', lambda e: app.quit())
        app.bind('<Control-n>', lambda e: safe_afficher_recherche())
        app.bind('<F5>', lambda e: safe_afficher_indices_boursiers())
        app.bind('<Escape>', lambda e: safe_clear_main_frame())

        try:
            if config_manager.is_openai_available():
                chat_btn = ctk.CTkButton(
                    app, text="💬", width=50, height=50, corner_radius=25,
                    font=("Segoe UI", 20), fg_color=COLORS['accent_blue'],
                    hover_color=COLORS['accent_green'],
                    command=lambda: safe_create_assistant_chat(app)
                )
                chat_btn.place(relx=0.97, rely=0.97, anchor="se")
                logger.info("Bouton chat créé")
        except Exception as e:
            logger.error(f"Erreur création bouton chat: {e}")

        safe_afficher_dashboard()
        logger.info("Application HelixOne démarrée avec succès")

    except Exception as e:
        logger.error(f"Erreur critique au démarrage: {e}")
        create_emergency_ui(app)


def create_fallback_sidebar(parent):
    """Crée une sidebar minimale en cas d'erreur"""
    sidebar = ctk.CTkFrame(parent, width=200, fg_color=COLORS['bg_secondary'])
    sidebar.pack_propagate(False)

    ctk.CTkLabel(
        sidebar, text="HelixOne", font=("Segoe UI", 20, "bold"),
        text_color=COLORS['accent_blue']
    ).pack(pady=30)

    ctk.CTkButton(
        sidebar, text="Dashboard", command=safe_afficher_dashboard
    ).pack(pady=10, padx=20, fill="x")

    return sidebar


def create_emergency_ui(app):
    """Interface d'urgence en cas d'erreur critique"""
    try:
        emergency_frame = ctk.CTkFrame(app, fg_color=COLORS['bg_primary'])
        emergency_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(
            emergency_frame, text="HelixOne - Mode Sécurisé",
            font=("Segoe UI", 24, "bold"), text_color=COLORS['accent_red']
        ).pack(pady=50)

        ctk.CTkLabel(
            emergency_frame,
            text="Une erreur critique s'est produite.\nConsultez les logs pour plus d'informations.",
            font=("Segoe UI", 14), text_color=COLORS['text_secondary']
        ).pack(pady=20)

        ctk.CTkButton(
            emergency_frame, text="Redémarrer", command=app.destroy
        ).pack(pady=30)
    except:
        pass


def safe_create_assistant_chat(parent):
    """Crée l'assistant chat de manière sécurisée"""
    try:
        SafeAssistantChatPopup(parent)
    except Exception as e:
        logger.error(f"Erreur création assistant chat: {e}")
        safe_show_notification("Erreur ouverture assistant chat", "error")


# ============================================================================
# PAGES SECURISEES
# ============================================================================
def safe_afficher_dashboard():
    """Dashboard sécurisé"""
    try:
        safe_clear_main_frame()

        title = ctk.CTkLabel(
            main_frame, text="Dashboard", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(0, 20))

        safe_afficher_indices_boursiers()
        safe_create_metric_cards()
        safe_create_ibkr_section()  # Section IBKR Portfolio
        logger.debug("Dashboard affiché avec succès")

    except Exception as e:
        logger.error(f"Erreur affichage dashboard: {e}")
        show_error_page("Erreur Dashboard", str(e))


def update_ibkr_metrics():
    """Mettre à jour les métriques avec les données IBKR"""
    def fetch_and_update():
        try:
            # Récupérer les données du microservice IBKR
            response = requests.get(f"{IBKR_MICROSERVICE_URL}/dashboard", timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('connected') and data.get('portfolio'):
                    portfolio = data['portfolio']

                    # Mettre à jour Portfolio (Net Liquidation)
                    if metric_cards['portfolio'] and metric_cards['portfolio'].winfo_exists():
                        net_liq = portfolio.get('net_liquidation', 0)
                        metric_cards['portfolio']._update_value_safe(f"{net_liq:,.2f} €")

                    # Mettre à jour P&L Jour (Unrealized PnL)
                    if metric_cards['pnl'] and metric_cards['pnl'].winfo_exists():
                        pnl = portfolio.get('unrealized_pnl', 0)
                        metric_cards['pnl']._update_value_safe(f"{pnl:,.2f} €")

                    # Mettre à jour Trades (nombre de positions)
                    if metric_cards['trades'] and metric_cards['trades'].winfo_exists():
                        positions = len(portfolio.get('positions', []))
                        metric_cards['trades']._update_value_safe(str(positions))

                    # Mettre à jour Win Rate (pour l'instant, statique ou calculé)
                    # TODO: Implémenter le calcul du win rate basé sur l'historique
                    if metric_cards['winrate'] and metric_cards['winrate'].winfo_exists():
                        # Pour l'instant, on affiche un placeholder
                        metric_cards['winrate']._update_value_safe("N/A")

                    logger.info("✅ Métriques IBKR mises à jour")
                else:
                    logger.warning("IBKR non connecté - métriques non mises à jour")
            else:
                logger.error(f"Erreur récupération dashboard IBKR: {response.status_code}")

        except requests.exceptions.ConnectionError:
            logger.warning("Microservice IBKR non disponible")
        except Exception as e:
            logger.error(f"Erreur mise à jour métriques IBKR: {e}")

    # Exécuter dans un thread pour ne pas bloquer l'UI
    thread = threading.Thread(target=fetch_and_update, daemon=True)
    thread.start()


def safe_create_metric_cards():
    """Cartes métriques sécurisées - Synchronisées avec IBKR"""
    global metric_cards

    try:
        cards_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        cards_frame.pack(fill="x", pady=20)

        # Valeurs par défaut (seront mises à jour par IBKR)
        cards_data = [
            ("Portfolio", 0, 0, "💼"),
            ("P&L Jour", 0, 0, "📊"),
            ("Trades", 0, 0, "🔄"),
            ("Win Rate", 0, 0, "🎯")
        ]

        card_keys = ['portfolio', 'pnl', 'trades', 'winrate']

        for i, (title, value, change, icon) in enumerate(cards_data):
            try:
                card = SafeMetricCard(cards_frame, title, value, change, icon)
                card.grid(row=0, column=i, padx=10, pady=5, sticky="ew")
                cards_frame.grid_columnconfigure(i, weight=1)

                # Stocker la référence
                metric_cards[card_keys[i]] = card
            except Exception as e:
                logger.error(f"Erreur création carte {title}: {e}")

        # Lancer la mise à jour des données IBKR
        update_ibkr_metrics()

    except Exception as e:
        logger.error(f"Erreur création cartes métriques: {e}")


def safe_create_ibkr_section():
    """Section IBKR Portfolio - Affiche le portefeuille en temps réel"""
    try:
        # Créer le panneau IBKR
        ibkr_panel = IBKRPortfolioPanel(main_frame, auth_token=None)
        ibkr_panel.pack(fill="both", expand=True, pady=20)

        logger.debug("Section IBKR Portfolio ajoutée au dashboard")

    except Exception as e:
        logger.error(f"Erreur création section IBKR: {e}")
        # Afficher un message d'erreur dans le dashboard
        error_frame = ctk.CTkFrame(
            main_frame, fg_color=COLORS['bg_secondary'], corner_radius=15
        )
        error_frame.pack(fill="x", pady=20)

        error_label = ctk.CTkLabel(
            error_frame,
            text=f"❌ Erreur chargement IBKR Portfolio: {str(e)}",
            font=("Segoe UI", 14),
            text_color=COLORS['accent_red']
        )
        error_label.pack(pady=30)


def safe_create_recent_section():
    """Section récente sécurisée"""
    try:
        recent_frame = ctk.CTkFrame(
            main_frame, fg_color=COLORS['bg_secondary'], corner_radius=15
        )
        recent_frame.pack(fill="both", expand=True, pady=20)

        recent_label = ctk.CTkLabel(
            recent_frame, text="Analyses Récentes", font=("Segoe UI", 18, "bold"),
            text_color=COLORS['text_primary']
        )
        recent_label.pack(pady=15)

        if data_manager.get_favorites():
            for ticker in data_manager.get_favorites()[:5]:
                ticker_label = ctk.CTkLabel(
                    recent_frame, text=f"📈 {ticker}", font=("Segoe UI", 12)
                )
                ticker_label.pack(pady=2)
        else:
            placeholder = ctk.CTkLabel(
                recent_frame, text="Aucune analyse récente", font=("Segoe UI", 14),
                text_color=COLORS['text_secondary']
            )
            placeholder.pack(pady=50)
    except Exception as e:
        logger.error(f"Erreur section récente: {e}")


def fetch_indices_data(labels: Dict, indices: Dict):
    """Récupère les données des indices avec threading sécurisé"""
    def update_single_index(ticker, name):
        try:
            current_minute = int(time.time() / 60)
            data = get_cached_ticker_data(ticker, current_minute)

            if data is not None and not data.empty:
                last = data["Close"].iloc[-1]
                change = ((last - data["Open"].iloc[0]) / data["Open"].iloc[0]) * 100
                color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
                text = f"{name}: {last:.2f} ({change:+.2f}%)"
                return ticker, text, color, True
            else:
                return ticker, f"{name}: Erreur", COLORS['accent_red'], False
        except Exception as e:
            logger.error(f"Erreur récupération données {ticker}: {e}")
            return ticker, f"{name}: Erreur", COLORS['accent_red'], False

    result_queue = queue.Queue()
    def worker():
        while True:
            try:
                ticker, name = result_queue.get(timeout=1)
                if ticker is None:
                    break

                result = update_single_index(ticker, name)

                def update_ui():
                    try:
                        ticker, text, color, success = result
                        if ticker in labels and labels[ticker].winfo_exists():
                            labels[ticker].configure(text=text, text_color=color)
                        if success:
                            logger.info(f"API call success: yfinance ticker/{ticker}")
                    except Exception as e:
                        logger.error(f"Erreur mise à jour interface indices: {e}")

                try:
                    if main_frame and main_frame.winfo_exists():
                        main_frame.after_idle(update_ui)
                except:
                    pass

                result_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Erreur worker thread: {e}")
                result_queue.task_done()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    for ticker, name in indices.items():
        result_queue.put((ticker, name))

    try:
        result_queue.join()
    except:
        pass

    result_queue.put((None, None))


def safe_afficher_indices_boursiers():
    """Indices boursiers sécurisés"""
    try:
        indices = {
            "^FCHI": "CAC 40",
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^DJI": "DOW JONES"
        }

        indices_frame = ctk.CTkFrame(
            main_frame, fg_color=COLORS['bg_secondary'], height=60, corner_radius=10
        )
        indices_frame.pack(fill="x", pady=(0, 20))

        labels = {}
        for i, (ticker, name) in enumerate(indices.items()):
            try:
                lbl = ctk.CTkLabel(
                    indices_frame, text=f"{name}: Chargement...",
                    text_color=COLORS['accent_blue'], font=("Segoe UI", 12)
                )
                lbl.grid(row=0, column=i, padx=15, pady=15)
                labels[ticker] = lbl
            except Exception as e:
                logger.error(f"Erreur création label {ticker}: {e}")

        def safe_fetch_wrapper():
            try:
                fetch_indices_data(labels, indices)
            except Exception as e:
                logger.error(f"Erreur fetch indices global: {e}")

        fetch_thread = threading.Thread(target=safe_fetch_wrapper, daemon=True)
        fetch_thread.start()

    except Exception as e:
        logger.error(f"Erreur lors de l'affichage des indices: {e}")


def safe_afficher_recherche():
    """Page recherche sécurisée"""
    try:
        global text_box, entry, suggestion_frame, notebook, tab_graphiques

        safe_clear_main_frame()

        title = ctk.CTkLabel(
            main_frame, text="Analyse de Marché", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(0, 20))

        safe_afficher_indices_boursiers()
        create_safe_search_zone()
        create_safe_results_tabs()

    except Exception as e:
        logger.error(f"Erreur page recherche: {e}")
        show_error_page("Erreur Recherche", str(e))


def create_safe_search_zone():
    """Zone de recherche sécurisée"""
    global entry, suggestion_frame, mode_var

    try:
        search_zone = ctk.CTkFrame(
            main_frame, corner_radius=15, fg_color=COLORS['bg_secondary']
        )
        search_zone.pack(padx=25, pady=25, fill="x")

        entry = ctk.CTkEntry(
            search_zone, width=400, height=45,
            placeholder_text="🔍 Rechercher une action (nom ou ticker)...",
            font=("Segoe UI", 14), fg_color=COLORS['bg_tertiary'],
            border_color=COLORS['border']
        )
        entry.pack(pady=20, padx=20)
        entry.bind("<KeyRelease>", safe_fetch_suggestions)

        suggestion_frame = ctk.CTkScrollableFrame(
            main_frame, height=80, fg_color=COLORS['bg_tertiary'], corner_radius=10
        )
        suggestion_frame.pack(padx=25, pady=(0, 10), fill="x")

        options_zone = ctk.CTkFrame(search_zone, fg_color="transparent")
        options_zone.pack(pady=(0, 20))

        try:
            ctk.CTkOptionMenu(
                options_zone, variable=mode_var,
                values=["Long Terme", "Court Terme", "Spéculatif"],
                fg_color=COLORS['bg_tertiary'], button_color=COLORS['accent_blue'],
                button_hover_color=COLORS['accent_green']
            ).grid(row=0, column=0, padx=10)

            ctk.CTkButton(
                options_zone, text="🔍 Analyser", font=("Segoe UI", 14, "bold"),
                fg_color=COLORS['accent_green'], hover_color=COLORS['accent_blue'],
                command=safe_analyser_action
            ).grid(row=0, column=1, padx=10)

            ctk.CTkButton(
                options_zone, text="📄 Export", font=("Segoe UI", 14),
                fg_color=COLORS['bg_tertiary'], hover_color=COLORS['bg_hover'],
                command=safe_exporter_rapport
            ).grid(row=0, column=2, padx=10)
        except Exception as e:
            logger.error(f"Erreur création boutons recherche: {e}")

    except Exception as e:
        logger.error(f"Erreur création zone recherche: {e}")


def create_safe_results_tabs():
    """Onglets de résultats sécurisés"""
    global notebook, text_box, tab_graphiques

    try:
        notebook = ctk.CTkTabview(main_frame, fg_color=COLORS['bg_secondary'])
        notebook.pack(padx=25, pady=10, fill="both", expand=True)

        # Onglet Analyse
        tab_analyse = notebook.add("🔍 Analyse")
        text_box = ctk.CTkTextbox(
            tab_analyse, font=("Consolas", 12), fg_color=COLORS['bg_tertiary'],
            text_color=COLORS['text_primary']
        )
        text_box.pack(fill="both", expand=True, padx=10, pady=10)
        text_box.insert("1.0", "Sélectionnez un ticker et cliquez sur 'Analyser' pour commencer.")

        # Onglet Graphiques
        tab_graphiques = notebook.add("📊 Graphiques")
        placeholder = ctk.CTkLabel(
            tab_graphiques, text="Les graphiques apparaîtront ici après l'analyse",
            font=("Segoe UI", 14), text_color=COLORS['text_secondary']
        )
        placeholder.pack(expand=True)

        # Onglet Communauté
        try:
            tab_communaute = notebook.add("💬 Communauté")
            container_chat = ctk.CTkFrame(tab_communaute, fg_color="transparent")
            container_chat.pack(fill="both", expand=True)

            if CommunityChat is None:
                ctk.CTkLabel(
                    container_chat, text="Module community_chat introuvable."
                ).pack(pady=20)
            else:
                chat = CommunityChat(container_chat, user_profile={"username": "Vous"})
                chat.pack(fill="both", expand=True)
        except Exception as e:
            try:
                ctk.CTkLabel(
                    tab_communaute, text=f"Erreur chargement du chat: {e}"
                ).pack(pady=10)
            except Exception:
                logger.exception("Erreur affichage message d'erreur dans l'onglet Communauté")

    except Exception as e:
        logger.error(f"Erreur création onglets résultats: {e}")


def safe_fetch_suggestions(event):
    """Récupère les suggestions de manière sécurisée"""
    try:
        if not entry or not suggestion_frame:
            return

        query = entry.get().strip()
        if not query:
            safe_show_suggestions([])
            return

        suggestions = data_manager.get_ticker_suggestions(query, 5)
        safe_show_suggestions(suggestions)
    except Exception as e:
        logger.error(f"Erreur fetch suggestions: {e}")


def safe_show_suggestions(suggestions: List[str]):
    """Affiche les suggestions de manière sécurisée"""
    try:
        if not suggestion_frame or not suggestion_frame.winfo_exists():
            return

        for widget in suggestion_frame.winfo_children():
            try:
                widget.destroy()
            except:
                continue

        for suggestion in suggestions[:5]:
            try:
                btn = ctk.CTkButton(
                    suggestion_frame, text=suggestion, height=30,
                    font=("Segoe UI", 12), fg_color=COLORS['bg_hover'],
                    text_color=COLORS['text_primary'], 
                    hover_color=COLORS['accent_blue'], anchor="w",
                    command=lambda s=suggestion: safe_on_suggestion_click(s)
                )
                btn.pack(fill="x", padx=5, pady=2)
            except Exception as e:
                logger.error(f"Erreur création suggestion {suggestion}: {e}")
    except Exception as e:
        logger.error(f"Erreur affichage suggestions: {e}")


def safe_on_suggestion_click(suggestion: str):
    """Gestion sécurisée du clic sur suggestion"""
    try:
        if entry and entry.winfo_exists():
            entry.delete(0, "end")
            entry.insert(0, suggestion)
            safe_show_suggestions([])
    except Exception as e:
        logger.error(f"Erreur clic suggestion: {e}")


def safe_analyser_action():
    """Analyse sécurisée d'une action avec nouveau moteur ML"""
    try:
        if not entry:
            safe_show_notification("Interface non initialisée", "error")
            return

        recherche = entry.get().strip()
        if not recherche:
            safe_show_notification("Veuillez entrer un ticker", "warning")
            return

        ticker = data_manager.find_ticker(recherche)
        if not ticker:
            ticker = recherche.upper()

        mode = mode_var.get() if mode_var else "Long Terme"
        # Mapper les modes français vers les modes API
        mode_map = {
            "Long Terme": "Conservative",
            "Court Terme": "Standard",
            "Spéculatif": "Aggressive"
        }
        api_mode = mode_map.get(mode, "Standard")

        safe_show_notification(f"Analyse ML de {ticker} en cours...", "info")

        if notebook:
            notebook.set("🔍 Analyse")

        # Afficher message de chargement dans l'onglet Analyse
        if text_box and text_box.winfo_exists():
            text_box.delete("1.0", "end")
            text_box.insert("end", "🤖 Analyse ML Enhanced en cours...\n\n")
            text_box.insert("end", "⏳ Collecte des données (35+ sources)...\n")
            text_box.insert("end", "🧠 Prédictions ML (XGBoost + LSTM)...\n")
            text_box.insert("end", "📊 Calcul Health Score...\n")
            text_box.insert("end", "💡 Génération recommandations...\n\n")

        # Créer container pour affichage ML dans l'onglet Analyse
        ml_display_container = None

        def run_ml_analysis():
            try:
                # Récupérer le token d'authentification
                from auth_session import get_auth_token
                token = get_auth_token()

                if not token:
                    logger.error("Pas de token d'authentification disponible")
                    raise Exception("Non authentifié. Veuillez vous reconnecter.")

                # Créer client API avec le token
                client = HelixOneClient()
                client.token = token
                logger.info(f"✅ Client authentifié pour analyse COMPLÈTE de {ticker}")

                # Appeler l'analyse ULTRA-COMPLÈTE (8 étapes - même que 2x/jour)
                try:
                    raw_result = client.deep_analyze(ticker)
                    logger.info(f"✅ Analyse complète 8 étapes reçue pour {ticker}")
                    use_deep_analysis = True
                except Exception as e:
                    logger.warning(f"⚠️ Deep analysis non disponible, fallback sur analyse standard: {e}")
                    raw_result = client.analyze(ticker, mode=api_mode)
                    use_deep_analysis = False

                # Adapter les champs du backend pour l'UI
                if use_deep_analysis:
                    # Structure de l'analyse COMPLÈTE (8 étapes)
                    ml_preds = raw_result.get('ml_predictions', {})
                    recommendation = raw_result.get('recommendation', {})
                    position_analysis = raw_result.get('position_analysis', {})

                    result = {
                        'use_deep_analysis': True,
                        'ticker': ticker,
                        'health_score': position_analysis.get('health_score', 50),
                        'recommendation_final': recommendation.get('action', 'HOLD'),
                        'confidence': recommendation.get('confidence', 50),
                        'score_technique': position_analysis.get('technical_score', 0),
                        'score_fondamental': position_analysis.get('fundamental_score', 0),
                        'score_sentiment': raw_result.get('sentiment_analysis', {}).get('sentiment_score', 0),
                        'score_risque': position_analysis.get('risk_score', 0),
                        'score_macro': 0,  # TODO: extraire du macro context
                        'score_fxi': position_analysis.get('health_score', 0),

                        # ML Predictions (XGBoost + LSTM)
                        'ml_predictions': {
                            'signal': ml_preds.get('signal', 'HOLD'),
                            'signal_strength': ml_preds.get('confidence', 50),
                            'prediction_1d': ml_preds.get('prediction_1d', {}).get('direction', 'N/A'),
                            'confidence_1d': ml_preds.get('prediction_1d', {}).get('confidence', 0),
                            'prediction_3d': ml_preds.get('prediction_3d', {}).get('direction', 'N/A'),
                            'confidence_3d': ml_preds.get('prediction_3d', {}).get('confidence', 0),
                            'prediction_7d': ml_preds.get('prediction_7d', {}).get('direction', 'N/A'),
                            'confidence_7d': ml_preds.get('prediction_7d', {}).get('confidence', 0),
                            'model_version': ml_preds.get('model_version', 'XGBoost+LSTM'),
                        },

                        # Nouvelles données de l'analyse complète
                        'data_collection': raw_result.get('data_collection', {}),
                        'sentiment_analysis': raw_result.get('sentiment_analysis', {}),
                        'position_analysis': position_analysis,
                        'recommendation': recommendation,
                        'alerts': raw_result.get('alerts', {}),
                        'upcoming_events': raw_result.get('upcoming_events', []),
                        'executive_summary': raw_result.get('executive_summary', ''),
                        'raw_data': raw_result
                    }
                    logger.info(f"✅ Résultats COMPLETS adaptés - Score: {result['health_score']}, Recommandation: {result['recommendation_final']}")
                else:
                    # Structure de l'analyse STANDARD (fallback)
                    result = {
                        'use_deep_analysis': False,
                        'health_score': raw_result.get('final_score', 50),
                        'recommendation_final': raw_result.get('recommendation', 'ATTENDRE'),
                        'confidence': raw_result.get('confidence', 50),
                        'score_technique': raw_result.get('technical_score', 0),
                        'score_fondamental': raw_result.get('fundamental_score', 0),
                        'score_sentiment': raw_result.get('sentiment_score', 0),
                        'score_risque': raw_result.get('risk_score', 0),
                        'score_macro': raw_result.get('macro_score', 0),
                        'score_fxi': raw_result.get('final_score', 0),
                        'ml_predictions': {
                            'signal': 'BUY' if raw_result.get('recommendation') == 'ACHETER' else 'SELL' if raw_result.get('recommendation') == 'VENDRE' else 'HOLD',
                            'signal_strength': raw_result.get('confidence', 50),
                            'prediction_1d': 'N/A',
                            'confidence_1d': 0,
                            'prediction_3d': 'N/A',
                            'confidence_3d': 0,
                            'prediction_7d': 'N/A',
                            'confidence_7d': 0,
                            'model_version': raw_result.get('details', {}).get('engine_version', 'v1.0') if isinstance(raw_result.get('details'), dict) else 'v1.0',
                        },
                        'details': raw_result.get('details', {}),
                        'raw_data': raw_result
                    }
                    logger.info(f"✅ Résultats STANDARD adaptés - Score: {result['health_score']}, Recommandation: {result['recommendation_final']}")

                def update_ui_with_ml():
                    try:
                        if notebook and notebook.winfo_exists():
                            # Clear l'onglet Analyse
                            tab_analyse = notebook.tab("🔍 Analyse")

                            # Détruire les anciens widgets
                            for widget in tab_analyse.winfo_children():
                                widget.destroy()

                            # Créer le composant approprié selon le type d'analyse
                            if result.get('use_deep_analysis', False):
                                # Analyse COMPLÈTE 8 étapes
                                from src.interface.deep_analysis_display import DeepAnalysisDisplay
                                ml_display = DeepAnalysisDisplay(tab_analyse)
                                ml_display.pack(fill="both", expand=True, padx=5, pady=5)
                                ml_display.display_results(result, ticker)
                                logger.info("✅ Affichage analyse COMPLÈTE (8 étapes)")
                            else:
                                # Analyse STANDARD (fallback)
                                ml_display = MLResultsDisplay(tab_analyse)
                                ml_display.pack(fill="both", expand=True, padx=10, pady=10)
                                ml_display.display_results(result, ticker)
                                logger.info("✅ Affichage analyse standard")

                            data_manager.add_favorite(ticker)
                            safe_show_notification(f"✅ Analyse ML de {ticker} terminée !", "success")
                    except Exception as e:
                        logger.error(f"Erreur mise à jour UI ML: {e}")
                        # Fallback vers affichage texte
                        if text_box and text_box.winfo_exists():
                            text_box.delete("1.0", "end")
                            text_box.insert("end", f"Résultats pour {ticker}:\n\n")
                            text_box.insert("end", f"Health Score: {result.get('health_score', 'N/A')}/100\n")
                            text_box.insert("end", f"Recommandation: {result.get('recommendation_final', 'N/A')}\n")
                            text_box.insert("end", f"Confiance: {result.get('confidence', 0):.0f}%\n\n")

                            ml_pred = result.get('ml_predictions', {})
                            if ml_pred:
                                text_box.insert("end", "Prédictions ML:\n")
                                text_box.insert("end", f"  1j: {ml_pred.get('prediction_1d')} ({ml_pred.get('confidence_1d', 0):.0f}%)\n")
                                text_box.insert("end", f"  3j: {ml_pred.get('prediction_3d')} ({ml_pred.get('confidence_3d', 0):.0f}%)\n")
                                text_box.insert("end", f"  7j: {ml_pred.get('prediction_7d')} ({ml_pred.get('confidence_7d', 0):.0f}%)\n")

                if main_frame and main_frame.winfo_exists():
                    main_frame.after_idle(update_ui_with_ml)

            except Exception as e:
                logger.error(f"Erreur analyse ML: {e}")

                def show_error():
                    try:
                        if text_box and text_box.winfo_exists():
                            text_box.delete("1.0", "end")
                            text_box.insert("end", f"❌ Erreur analyse ML: {str(e)}\n\n")
                            text_box.insert("end", "Vérifiez:\n")
                            text_box.insert("end", "- Backend lancé (port 8000)\n")
                            text_box.insert("end", "- Authentification valide\n")
                            text_box.insert("end", "- Connexion internet\n")
                        safe_show_notification("Erreur lors de l'analyse ML", "error")
                    except:
                        pass

                if main_frame and main_frame.winfo_exists():
                    main_frame.after_idle(show_error)

        threading.Thread(target=run_ml_analysis, daemon=True).start()

    except Exception as e:
        logger.error(f"Erreur analyser_action: {e}")
        safe_show_notification("Erreur lors du lancement de l'analyse", "error")


def safe_exporter_rapport():
    """Export sécurisé du rapport"""
    try:
        if not text_box or not text_box.winfo_exists():
            safe_show_notification("Aucun rapport à exporter", "warning")
            return

        rapport = text_box.get("1.0", "end").strip()
        if not rapport or len(rapport) < 10:
            safe_show_notification("Aucun contenu à exporter", "warning")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")],
            title="Exporter le rapport"
        )

        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(rapport)
                safe_show_notification("Rapport exporté avec succès", "success")
                logger.info(f"Rapport exporté: {filepath}")
            except Exception as e:
                logger.error(f"Erreur écriture fichier: {e}")
                safe_show_notification("Erreur lors de l'export", "error")

    except Exception as e:
        logger.error(f"Erreur export rapport: {e}")
        safe_show_notification("Erreur lors de l'export", "error")


def safe_afficher_graphiques():
    """Page graphiques sécurisée - ULTRA PROFESSIONNEL 🔥"""
    try:
        safe_clear_main_frame()

        # Import du nouveau système de graphiques avancés
        from src.interface.advanced_charts_panel import afficher_advanced_charts

        # Afficher le panel professionnel
        afficher_advanced_charts(main_frame)

        logger.info("✅ Advanced Charts Panel affiché avec succès")

    except Exception as e:
        logger.error(f"Erreur page graphiques: {e}", exc_info=True)
        show_error_page("Erreur Graphiques", str(e))


def safe_afficher_portfolio():
    """Page portfolio avec analyse ML sécurisée"""
    try:
        safe_clear_main_frame()

        title = ctk.CTkLabel(
            main_frame, text="📊 Mon Portfolio - Analyse ML", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(0, 20))

        # Créer le client API avec authentification
        try:
            client = HelixOneClient()

            # Récupérer le token d'authentification
            if auth_manager_global and hasattr(auth_manager_global, 'token'):
                client.token = auth_manager_global.token
                logger.info("Token d'authentification récupéré pour le portfolio")
            else:
                logger.warning("Aucun token d'authentification disponible")
                safe_show_notification("Authentification requise", "warning")
                return

            # Créer le panel d'analyse de portfolio
            portfolio_panel = PortfolioAnalysisPanel(main_frame, api_client=client)
            portfolio_panel.pack(fill="both", expand=True, padx=20, pady=10)

            logger.info("✅ Panel Portfolio ML affiché")

        except ImportError as e:
            logger.error(f"Module PortfolioAnalysisPanel non trouvé: {e}")
            show_error_page("Module Portfolio Introuvable",
                          "Le module d'analyse portfolio n'est pas disponible.\nVérifiez l'installation.")
        except Exception as e:
            logger.error(f"Erreur création panel portfolio: {e}")
            show_error_page("Erreur Portfolio", str(e))

    except Exception as e:
        logger.error(f"Erreur page portfolio: {e}")
        show_error_page("Erreur Portfolio", str(e))


def safe_afficher_zone_alerte():
    """Page alertes sécurisée"""
    try:
        safe_clear_main_frame()

        title = ctk.CTkLabel(
            main_frame, text="Centre d'Alertes", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(0, 20))

        alert_frame = ctk.CTkFrame(
            main_frame, fg_color=COLORS['bg_secondary'], corner_radius=15
        )
        alert_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            alert_frame, text="🎯 Système d'alertes en développement",
            font=("Segoe UI", 16), text_color=COLORS['text_secondary']
        ).pack(expand=True)

    except Exception as e:
        logger.error(f"Erreur page alertes: {e}")
        show_error_page("Erreur Alertes", str(e))


def safe_afficher_parametres():
    """Page paramètres sécurisée"""
    try:
        safe_clear_main_frame()

        title = ctk.CTkLabel(
            main_frame, text="Paramètres", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(0, 20))

        # Container des paramètres
        settings_container = ctk.CTkScrollableFrame(
            main_frame, fg_color=COLORS['bg_secondary']
        )
        settings_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Section Apparence
        section_app = ctk.CTkFrame(settings_container, fg_color=COLORS['bg_tertiary'])
        section_app.pack(fill="x", pady=10, padx=20)

        ctk.CTkLabel(
            section_app, text="🎨 Apparence", font=("Segoe UI", 16, "bold"), 
            text_color=COLORS['text_primary']
        ).pack(pady=10)

        theme_default = "Light" if getattr(config_manager.config, 'theme', 'dark').lower() == "light" else "Dark"
        lang_default = "English" if getattr(config_manager.config, 'language', 'fr').lower().startswith("en") else "Français"

        # Thème
        row_app = ctk.CTkFrame(section_app, fg_color="transparent")
        row_app.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_app, text="Thème", font=("Segoe UI", 12), text_color=COLORS['text_secondary']).pack(side="left", padx=10)
        app_state.settings_vars["theme"] = ctk.StringVar(value=theme_default)
        ctk.CTkOptionMenu(
            row_app, variable=app_state.settings_vars["theme"],
            values=["Dark", "Light"], fg_color=COLORS['bg_hover'], 
            button_color=COLORS['accent_blue']
        ).pack(side="right", padx=10)

        # Langue
        row_lang = ctk.CTkFrame(section_app, fg_color="transparent")
        row_lang.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_lang, text="Langue", font=("Segoe UI", 12), text_color=COLORS['text_secondary']).pack(side="left", padx=10)
        app_state.settings_vars["langue"] = ctk.StringVar(value=lang_default)
        ctk.CTkOptionMenu(
            row_lang, variable=app_state.settings_vars["langue"],
            values=["Français", "English"], fg_color=COLORS['bg_hover'], 
            button_color=COLORS['accent_blue']
        ).pack(side="right", padx=10)

        # Section Trading
        section_tr = ctk.CTkFrame(settings_container, fg_color=COLORS['bg_tertiary'])
        section_tr.pack(fill="x", pady=10, padx=20)

        ctk.CTkLabel(
            section_tr, text="📊 Trading", font=("Segoe UI", 16, "bold"), 
            text_color=COLORS['text_primary']
        ).pack(pady=10)

        mode_default = user_config.get("mode", "Long Terme")
        alerts_default = "Oui" if user_config.get("alerte_active", True) else "Non"

        # Mode par défaut
        row_mode = ctk.CTkFrame(section_tr, fg_color="transparent")
        row_mode.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_mode, text="Mode par défaut", font=("Segoe UI", 12), text_color=COLORS['text_secondary']).pack(side="left", padx=10)
        app_state.settings_vars["mode"] = ctk.StringVar(value=mode_default)
        ctk.CTkOptionMenu(
            row_mode, variable=app_state.settings_vars["mode"],
            values=["Long Terme", "Court Terme", "Spéculatif"], 
            fg_color=COLORS['bg_hover'], button_color=COLORS['accent_blue']
        ).pack(side="right", padx=10)

        # Alertes
        row_alert = ctk.CTkFrame(section_tr, fg_color="transparent")
        row_alert.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_alert, text="Alertes actives", font=("Segoe UI", 12), text_color=COLORS['text_secondary']).pack(side="left", padx=10)
        app_state.settings_vars["alerte_active"] = ctk.StringVar(value=alerts_default)
        ctk.CTkOptionMenu(
            row_alert, variable=app_state.settings_vars["alerte_active"],
            values=["Oui", "Non"], fg_color=COLORS['bg_hover'], 
            button_color=COLORS['accent_blue']
        ).pack(side="right", padx=10)

        # Section API
        api_frame = ctk.CTkFrame(settings_container, fg_color=COLORS['bg_tertiary'])
        api_frame.pack(fill="x", pady=20, padx=20)

        ctk.CTkLabel(
            api_frame, text="🔑 Configuration API", font=("Segoe UI", 16, "bold"), 
            text_color=COLORS['text_primary']
        ).pack(pady=10)

        if config_manager.is_openai_available():
            status_text = "✅ OpenAI API configurée"
            status_color = COLORS['accent_green']
        else:
            status_text = "⚠ OpenAI API non configurée"
            status_color = COLORS['accent_red']

        ctk.CTkLabel(api_frame, text=status_text, font=("Segoe UI", 12), text_color=status_color).pack(pady=5)

        # Bouton sauvegarder
        ctk.CTkButton(
            settings_container, text="💾 Sauvegarder les paramètres",
            font=("Segoe UI", 14, "bold"), fg_color=COLORS['accent_green'],
            hover_color=COLORS['accent_blue'], command=save_settings
        ).pack(pady=30)

    except Exception as e:
        logger.error(f"Erreur page paramètres: {e}")
        show_error_page("Erreur Paramètres", str(e))


def safe_afficher_communaute():
    """Page Communauté (chat)"""
    try:
        safe_clear_main_frame()
        container = ctk.CTkFrame(main_frame, fg_color=COLORS['bg_primary'])
        container.pack(fill="both", expand=True)
        
        title = ctk.CTkLabel(
            container, text="💬 Communauté", font=("Segoe UI", 24, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(20, 10))
        
        body = ctk.CTkFrame(container, fg_color=COLORS['bg_secondary'])
        body.pack(fill="both", expand=True, padx=20, pady=20)
        
        if CommunityChat is None:
            ctk.CTkLabel(
                body, text="Module community_chat introuvable.",
                font=("Segoe UI", 14), text_color=COLORS['text_secondary']
            ).pack(pady=20)
        else:
            chat = CommunityChat(body, user_profile={"username": "Vous"})
            chat.pack(fill="both", expand=True)
    except Exception as e:
        logger.error(f"Erreur page Communauté: {e}")
        show_error_page("Erreur Communauté", str(e))


def save_settings():
    """Sauvegarde les paramètres"""
    try:
        if not app_state.settings_vars:
            safe_show_notification("Interface paramètres non initialisée", "error")
            return

        # Lire valeurs UI
        theme_ui = app_state.settings_vars["theme"].get()
        langue_ui = app_state.settings_vars["langue"].get()
        mode_ui = app_state.settings_vars["mode"].get()
        alert_ui = app_state.settings_vars["alerte_active"].get()

        # Normalisation
        theme_cfg = theme_ui.lower()
        lang_cfg = "fr" if langue_ui.lower().startswith("fr") else "en"
        alert_bool = True if alert_ui == "Oui" else False

        # Charger config existante
        data = {}
        if config_manager._config_path.exists():
            try:
                with open(config_manager._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data.setdefault("openai_api_key", getattr(config_manager.config, 'openai_api_key', ''))
        data.setdefault("app_settings", {})
        data["app_settings"]["theme"] = theme_cfg
        data["app_settings"]["language"] = lang_cfg
        if "debug" not in data["app_settings"]:
            data["app_settings"]["debug"] = getattr(config_manager.config, 'debug_mode', False)
        data.setdefault("api_settings", {
            "timeout": getattr(config_manager.config, 'timeout', 30),
            "retries": getattr(config_manager.config, 'max_retries', 3)
        })

        with open(config_manager._config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Mettre à jour en mémoire
        if hasattr(config_manager.config, 'theme'):
            config_manager.config.theme = theme_cfg
        if hasattr(config_manager.config, 'language'):
            config_manager.config.language = lang_cfg

        user_config["theme"] = theme_cfg
        user_config["langue"] = "Français" if lang_cfg == "fr" else "English"
        user_config["mode"] = mode_ui
        user_config["alerte_active"] = alert_bool

        # Appliquer le thème
        _apply_runtime_theme(theme_cfg)

        # Synchroniser la variable globale
        global mode_var
        if mode_var:
            try:
                mode_var.set(mode_ui)
            except Exception:
                pass

        safe_show_notification("Paramètres sauvegardés ✅", "success")
        logger.info(f"Paramètres sauvegardés: theme={theme_cfg}, langue={lang_cfg}, mode={mode_ui}, alertes={alert_bool}")

    except Exception as e:
        logger.error(f"Erreur sauvegarde paramètres: {e}")
        safe_show_notification("Erreur lors de la sauvegarde", "error")


def show_error_page(title: str, message: str):
    """Affiche une page d'erreur"""
    try:
        safe_clear_main_frame()

        error_container = ctk.CTkFrame(
            main_frame, fg_color=COLORS['bg_secondary'], corner_radius=15
        )
        error_container.pack(expand=True, padx=50, pady=50)

        ctk.CTkLabel(
            error_container, text="⚠", font=("Segoe UI", 48)
        ).pack(pady=20)

        ctk.CTkLabel(
            error_container, text=title, font=("Segoe UI", 20, "bold"),
            text_color=COLORS['accent_red']
        ).pack(pady=10)

        ctk.CTkLabel(
            error_container, text=message[:200], font=("Segoe UI", 12),
            text_color=COLORS['text_secondary'], wraplength=400
        ).pack(pady=10, padx=20)

        ctk.CTkButton(
            error_container, text="Retour au Dashboard",
            command=safe_afficher_dashboard, fg_color=COLORS['accent_blue']
        ).pack(pady=20)

    except Exception as e:
        logger.critical(f"Erreur critique affichage erreur: {e}")


# ============================================================================
# ASSISTANT CHAT SECURISE
# ============================================================================
class SafeAssistantChatPopup(ctk.CTkToplevel):
    """Assistant chat avec gestion d'erreurs"""

    def __init__(self, parent):
        try:
            super().__init__(parent)
            self.title("Assistant IA HelixOne")
            self.geometry("450x600")
            self.configure(fg_color=COLORS['bg_primary'])
            self.resizable(False, False)

            if not config_manager.is_openai_available():
                self.show_no_api_message()
                return

            self.setup_ui()
            self.chat_history = []
            self.add_message(
                "assistant",
                "Bonjour! Je suis votre assistant trading HelixOne. "
                "Comment puis-je vous aider aujourd'hui?"
            )

        except Exception as e:
            logger.error(f"Erreur création chat: {e}")
            self.destroy()

    def setup_ui(self):
        """Configure l'interface du chat"""
        # Header
        header = ctk.CTkFrame(self, fg_color=COLORS['bg_secondary'], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🤖 Assistant Trading IA", font=("Segoe UI", 16, "bold"),
            text_color=COLORS['text_primary']
        ).pack(expand=True)

        # Zone de chat
        self.chat_frame = ctk.CTkScrollableFrame(
            self, fg_color=COLORS['bg_tertiary']
        )
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Zone d'entrée
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=10)

        self.input_entry = ctk.CTkEntry(
            input_frame, placeholder_text="Tapez votre message...",
            height=40, font=("Segoe UI", 12), fg_color=COLORS['bg_secondary'],
            border_color=COLORS['border']
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", lambda e: self.send_message())

        send_btn = ctk.CTkButton(
            input_frame, text="➤", width=40, height=40, font=("Segoe UI", 16),
            fg_color=COLORS['accent_blue'], hover_color=COLORS['accent_green'],
            command=self.send_message
        )
        send_btn.pack(side="right")

    def show_no_api_message(self):
        """Message quand l'API n'est pas configurée"""
        frame = ctk.CTkFrame(self, fg_color=COLORS['bg_secondary'])
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame, text="⚠️ API OpenAI non configurée",
            font=("Segoe UI", 16, "bold"), text_color=COLORS['accent_yellow']
        ).pack(pady=20)

        ctk.CTkLabel(
            frame,
            text="Pour utiliser l'assistant IA, ajoutez votre clé API\ndans le fichier config.json",
            font=("Segoe UI", 12), text_color=COLORS['text_secondary'],
            wraplength=350
        ).pack(pady=10)

        ctk.CTkButton(frame, text="Fermer", command=self.destroy).pack(pady=20)

    def add_message(self, sender: str, message: str):
        """Ajoute un message au chat"""
        try:
            msg_frame = ctk.CTkFrame(
                self.chat_frame,
                fg_color=COLORS['bg_hover'] if sender == "user" else COLORS['bg_secondary'],
                corner_radius=10
            )
            msg_frame.pack(fill="x", pady=5, padx=10)

            # Icône et nom
            header_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=(5, 0))

            icon = "👤" if sender == "user" else "🤖"
            name = "Vous" if sender == "user" else "Assistant"
            color = COLORS['accent_blue'] if sender == "user" else COLORS['accent_green']

            ctk.CTkLabel(
                header_frame, text=f"{icon} {name}",
                font=("Segoe UI", 11, "bold"), text_color=color
            ).pack(anchor="w")

            # Message
            msg_label = ctk.CTkLabel(
                msg_frame, text=message[:500], font=("Segoe UI", 11),
                text_color=COLORS['text_primary'], wraplength=380, justify="left"
            )
            msg_label.pack(anchor="w", padx=10, pady=(5, 10))

            # Scroll vers le bas
            self.chat_frame._parent_canvas.yview_moveto(1.0)

        except Exception as e:
            logger.error(f"Erreur ajout message: {e}")

    def send_message(self):
        """Envoie un message"""
        try:
            message = self.input_entry.get().strip()
            if not message:
                return

            self.add_message("user", message)
            self.chat_history.append({"role": "user", "content": message})
            self.input_entry.delete(0, "end")
            self.add_message("assistant", "En train d'écrire...")

            threading.Thread(
                target=self.get_ai_response, args=(message,), daemon=True
            ).start()

        except Exception as e:
            logger.error(f"Erreur envoi message: {e}")
            self.add_message("assistant", "Désolé, une erreur s'est produite.")

    def get_ai_response(self, message: str):
        """Obtient la réponse de l'IA"""
        try:
            if not config_manager.client:
                response = "L'API OpenAI n'est pas configurée."
            else:
                system_prompt = (
                    "Tu es un assistant expert en trading et analyse financière. "
                    "Tu aides avec des analyses techniques, des stratégies de trading, "
                    "et des conseils sur les marchés financiers. Sois précis et professionnel."
                )

                messages = [
                    {"role": "system", "content": system_prompt}
                ] + self.chat_history[-10:]

                completion = config_manager.client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=messages,
                    temperature=0.7, max_tokens=500
                )
                response = completion.choices[0].message.content

            self.after(0, self.update_ai_response, response)

        except Exception as e:
            logger.error(f"Erreur API OpenAI: {e}")
            self.after(0, self.update_ai_response,
                      "Désolé, je ne peux pas répondre pour le moment.")

    def update_ai_response(self, response: str):
        """Met à jour la réponse de l'IA dans l'UI"""
        try:
            if self.chat_frame.winfo_children():
                self.chat_frame.winfo_children()[-1].destroy()

            self.add_message("assistant", response)
            self.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            logger.error(f"Erreur mise à jour réponse: {e}")


# ============================================================================
# FONCTION D'EXPORT
# ============================================================================
__all__ = [
    'launch_main_app',
    'safe_show_notification',
    'config_manager',
    'data_manager',
    'logger'
]