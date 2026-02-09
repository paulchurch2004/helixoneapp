"""
HelixOne - Main Application (Formation Only)
"""
from PIL import Image
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from src.interface.formation_commerciale import afficher_formation_commerciale
from src.interface.toast_notification import ToastManager
from src.interface.profile_panel import ProfilePanel
from src.interface.tooltips import Tooltip, add_tooltip
from src.interface.ui_components import EmptyState, WelcomeNotification
from src.interface.design_system import (
    DESIGN_TOKENS,
    COLORS as DS_COLORS,
    GLASS as DS_GLASS,
    SPACING as DS_SPACING,
    ANIMATION as DS_ANIMATION,
    GLASS_PANEL_STYLE, GLASS_BUTTON_STYLE, NAV_BUTTON_STYLE, NAV_BUTTON_ACTIVE_STYLE,
    SIDEBAR_STYLE, ICONS, get_color, get_glass, get_spacing
)
from tkinter import messagebox
from typing import Dict, List, Optional, Any, Union, Type
import customtkinter as ctk
import json
import logging
import os
import sys
import threading
import time
import tkinter as tk

# Import de la configuration centralisee
from src.config import get_api_url

# Import du client API
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.helixone_client import HelixOneClient

# Import du système de traduction i18n
from src.i18n import t, set_language, LanguagePreferences


# ============================================================================
# SYSTEME DE LOGGING
# ============================================================================
def setup_logging() -> logging.Logger:
    """Configure le système de logging"""
    # En mode packagé, écrire les logs dans ~/Library/Logs/HelixOne/
    if getattr(sys, 'frozen', False):
        log_dir = Path.home() / "Library" / "Logs" / "HelixOne"
    else:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

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
# IMPORTS DES EFFETS VISUELS
# ============================================================================
PREMIUM_EFFECTS_AVAILABLE = False

try:
    from src.interface.glassmorphism import (
        GlassFrame, GlassPanel, GlassCard,
        GlassButton, FloatingPanel
    )
    from src.interface.chart_animations import (
        AnimatedBarChart, AnimatedLineChart,
        ProgressBar, DonutChart, SparklineChart
    )
    from src.interface.theme_manager import theme_manager, AnimatedThemeTransition

    class ParticleCanvas:
        def __init__(self, *args, **kwargs): pass
        def create(self): return None
        def stop(self): pass

    class TypewriterLabel(ctk.CTkLabel):
        def __init__(self, parent, full_text="", *args, **kwargs):
            kwargs.pop('speed', None)
            super().__init__(parent, text=full_text, *args, **kwargs)
        def start_typing(self): pass

    class PulsingButton(ctk.CTkButton):
        def start_pulse(self): pass

    PREMIUM_EFFECTS_AVAILABLE = True
    logger.info("Effets visuels chargés")

except ImportError as e:
    PREMIUM_EFFECTS_AVAILABLE = False
    logger.warning(f"Modules visuels non disponibles: {e}")

    class ParticleCanvas:
        def __init__(self, *args, **kwargs): pass
        def create(self): return None
        def stop(self): pass

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
    theme: str = "dark"
    language: str = "fr"
    timeout: int = 30
    max_retries: int = 3
    debug_mode: bool = False


class ConfigManager:
    """Gestionnaire de configuration"""

    def __init__(self):
        self._config_path = Path("data/config.json")
        self.config = self._load_config()

    def _load_config(self):
        """Charge la configuration"""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                app_settings = data.get("app_settings", {})
                api_settings = data.get("api_settings", {})

                return AppConfig(
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
                if key in ["theme", "language", "debug"]:
                    data.setdefault("app_settings", {})[key] = value
                elif key in ["timeout", "retries"]:
                    data.setdefault("api_settings", {})[key] = value

            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.config = self._load_config()

        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")


# === INSTANCES GLOBALES ===
config_manager = ConfigManager()

# Définir THEMES au niveau global
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

# Configuration du thème
if PREMIUM_EFFECTS_AVAILABLE:
    initial_theme = theme_manager.get_current_theme()
    COLORS = initial_theme['colors']
else:
    initial_theme_name = getattr(config_manager.config, 'theme', 'dark')
    COLORS = dict(THEMES[initial_theme_name])

user_config = {
    "theme": getattr(config_manager.config, 'theme', 'dark'),
    "langue": "Français",
    "alerte_active": True
}


# ============================================================================
# PULSING GLOW ICON (used by sidebar error)
# ============================================================================
class PulsingGlowIcon(tk.Canvas):
    """Icône avec effet de pulsation"""
    ICON_PATHS = {
        'alerts': [(0.5, 0.2), (0.2, 0.8), (0.8, 0.8)],
        'error': [(0.3, 0.3), (0.7, 0.7), (0.7, 0.3), (0.3, 0.7)],
    }

    def __init__(self, parent, icon_type='alerts', size=24, color='#00D4FF',
                 glow=True, bg='#0d1117'):
        super().__init__(parent, width=size, height=size, bg=bg,
                         highlightthickness=0, bd=0)
        self.size = size
        self.color = color
        self.glow = glow
        self.icon_type = icon_type
        self._draw_icon()

    def _draw_icon(self):
        s = self.size
        points = self.ICON_PATHS.get(self.icon_type, self.ICON_PATHS['alerts'])
        scaled = [(x * s, y * s) for x, y in points]
        if len(scaled) >= 3:
            flat = [coord for point in scaled for coord in point]
            self.create_polygon(flat, fill=self.color, outline=self.color)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_modifier_key():
    """Retourne le symbole du modificateur (Cmd sur Mac, Ctrl ailleurs)"""
    return "⌘" if sys.platform == "darwin" else "Ctrl"

def get_binding_modifier():
    """Retourne le nom du binding (Command sur Mac, Control ailleurs)"""
    return "Command" if sys.platform == "darwin" else "Control"


# ============================================================================
# MODERN SIDEBAR
# ============================================================================
class ModernSidebar(ctk.CTkFrame):
    """Sidebar professionnelle moderne avec collapse/expand"""

    SIDEBAR_COLORS = {
        'bg': DS_COLORS['bg_primary'],
        'bg_glass': DS_GLASS['bg_dark'],
        'bg_secondary': DS_COLORS['bg_secondary'],
        'bg_hover': DS_GLASS['bg'],
        'bg_active': DS_GLASS['bg_light'],
        'border': DS_GLASS['border'],
        'border_active': DS_GLASS['border_active'],
        'text': DS_COLORS['text_primary'],
        'text_muted': DS_COLORS['text_secondary'],
        'accent': DS_COLORS['accent_cyan'],
        'accent_glow': DS_GLASS['glow'],
        'danger': DS_COLORS['error'],
        'danger_hover': '#ff4757',
    }

    def __init__(self, parent, main_frame, user_info=None):
        super().__init__(
            parent,
            width=240,
            fg_color=self.SIDEBAR_COLORS['bg_glass'],
            border_width=1,
            border_color=self.SIDEBAR_COLORS['border']
        )
        self.parent = parent
        self.main_frame = main_frame
        self.user_info = user_info or {}
        self.pack_propagate(False)

        self.buttons = []
        self.is_collapsed = False
        self.sidebar_width = 240
        self.collapsed_width = 60
        self._active_index = 0
        self._collapsible_widgets = []

        try:
            self._build_sidebar()
            self.select_button(0)
        except Exception as e:
            logger.error(f"Erreur création Sidebar: {e}")
            self._create_error_sidebar()

    def _build_sidebar(self):
        """Construit la sidebar complète"""

        # HEADER
        self.header = ctk.CTkFrame(self, fg_color="transparent", height=56)
        self.header.pack(fill="x", padx=12, pady=(12, 8))
        self.header.pack_propagate(False)

        self.toggle_btn = ctk.CTkButton(
            self.header,
            text="≡",
            width=36, height=36,
            font=("Arial", 20, "bold"),
            fg_color=self.SIDEBAR_COLORS['bg_hover'],
            hover_color=self.SIDEBAR_COLORS['border'],
            text_color=self.SIDEBAR_COLORS['text'],
            corner_radius=8,
            command=self.toggle_sidebar
        )
        self.toggle_btn.pack(side="left")

        self.logo_container = ctk.CTkFrame(self.header, fg_color="transparent")
        self.logo_container.pack(side="left", padx=(12, 0), fill="y")
        self._collapsible_widgets.append(self.logo_container)

        logo_inner = ctk.CTkFrame(self.logo_container, fg_color="transparent")
        logo_inner.pack(expand=True)

        ctk.CTkLabel(
            logo_inner, text="Helix",
            font=("Segoe UI Semibold", 18), text_color="#ffffff"
        ).pack(side="left")
        ctk.CTkLabel(
            logo_inner, text="One",
            font=("Segoe UI Semibold", 18), text_color=self.SIDEBAR_COLORS['accent']
        ).pack(side="left")

        # SEPARATOR
        sep = ctk.CTkFrame(self, height=1, fg_color=self.SIDEBAR_COLORS['bg_hover'])
        sep.pack(fill="x", padx=12, pady=(4, 12))

        # NAVIGATION MENU
        self.nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.nav_frame.pack(fill="both", expand=True, padx=8)

        mod = get_modifier_key()
        menu_items = [
            ("Formation", ICONS.get('info', '\u2139'), self.show_formation, f"{mod}+1",
             "Apprenez le trading avec notre Academy"),
        ]

        for idx, (text, icon, command, shortcut, tooltip) in enumerate(menu_items):
            btn = self._create_nav_button(text, icon, command, idx, shortcut, tooltip)
            btn.pack(fill="x", pady=2)
            self.buttons.append(btn)

        self._nav_commands = [item[2] for item in menu_items]
        self._bind_keyboard_shortcuts()

        # FOOTER
        self.footer = ctk.CTkFrame(
            self,
            fg_color=self.SIDEBAR_COLORS['bg_hover'],
            corner_radius=DS_GLASS['radius'],
            border_width=1,
            border_color=self.SIDEBAR_COLORS['border']
        )
        self.footer.pack(fill="x", padx=8, pady=(8, 12))

        # Profil Row
        self.profile_row = ctk.CTkFrame(self.footer, fg_color="transparent", height=44, corner_radius=8)
        self.profile_row.pack(fill="x", padx=6, pady=(8, 4))
        self.profile_row.pack_propagate(False)

        first = self.user_info.get('first_name', '')
        last = self.user_info.get('last_name', '')
        initials = f"{first[0] if first else ''}{last[0] if last else ''}".upper() or "U"

        self.avatar = ctk.CTkFrame(self.profile_row, width=32, height=32,
                                    fg_color=self.SIDEBAR_COLORS['accent'], corner_radius=16)
        self.avatar.pack(side="left", padx=(8, 0), pady=6)
        self.avatar.pack_propagate(False)
        avatar_label = ctk.CTkLabel(self.avatar, text=initials,
                                     font=("Segoe UI", 11, "bold"), text_color="#000")
        avatar_label.pack(expand=True)

        self.profile_label = ctk.CTkLabel(
            self.profile_row, text="Mon Profil",
            font=("Segoe UI", 12), text_color=self.SIDEBAR_COLORS['text'], anchor="w"
        )
        self.profile_label.pack(side="left", padx=(10, 0), fill="y")
        self._collapsible_widgets.append(self.profile_label)

        self.profile_chevron = ctk.CTkLabel(
            self.profile_row, text=">",
            font=("Segoe UI", 12), text_color=self.SIDEBAR_COLORS['text_muted']
        )
        self.profile_chevron.pack(side="right", padx=(0, 10))
        self._collapsible_widgets.append(self.profile_chevron)

        def on_profile_enter(e):
            self.profile_row.configure(fg_color=self.SIDEBAR_COLORS['bg_hover'])
        def on_profile_leave(e):
            self.profile_row.configure(fg_color="transparent")
        def on_profile_click(e):
            self.show_profile()

        for widget in [self.profile_row, self.avatar, avatar_label, self.profile_label, self.profile_chevron]:
            widget.bind("<Enter>", on_profile_enter)
            widget.bind("<Leave>", on_profile_leave)
            widget.bind("<Button-1>", on_profile_click)
            try:
                widget.configure(cursor="hand2")
            except Exception:
                pass

        # Actions Row
        self.actions_row = ctk.CTkFrame(self.footer, fg_color="transparent", height=40)
        self.actions_row.pack(fill="x", padx=6, pady=(4, 8))
        self.actions_row.pack_propagate(False)

        self.settings_btn = ctk.CTkButton(
            self.actions_row,
            text="Parametres",
            height=34,
            font=("Segoe UI", 11),
            fg_color=self.SIDEBAR_COLORS['bg_hover'],
            hover_color=self.SIDEBAR_COLORS['border'],
            text_color=self.SIDEBAR_COLORS['text_muted'],
            corner_radius=8,
            command=self.show_settings
        )
        self.settings_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._collapsible_widgets.append(self.settings_btn)

        self.logout_btn = ctk.CTkButton(
            self.actions_row,
            text="X",
            width=34, height=34,
            font=("Segoe UI", 12, "bold"),
            fg_color=self.SIDEBAR_COLORS['bg_hover'],
            hover_color=self.SIDEBAR_COLORS['danger_hover'],
            text_color=self.SIDEBAR_COLORS['danger'],
            corner_radius=8,
            command=self._do_logout
        )
        self.logout_btn.pack(side="right")

    def _create_nav_button(self, text, icon, command, index, shortcut="", tooltip=""):
        """Crée un bouton de navigation"""
        frame = ctk.CTkFrame(
            self.nav_frame, fg_color="transparent", height=44,
            corner_radius=DS_GLASS['radius_sm']
        )
        frame.pack_propagate(False)

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=4)

        indicator = ctk.CTkFrame(inner, width=3, fg_color="transparent", corner_radius=2)
        indicator.pack(side="left", fill="y", pady=6)

        icon_label = ctk.CTkLabel(
            inner, text=icon, font=("Segoe UI", 14),
            text_color=self.SIDEBAR_COLORS['text_muted'], width=28
        )
        icon_label.pack(side="left", padx=(8, 0))

        text_label = ctk.CTkLabel(
            inner, text=text, font=("Segoe UI", 12),
            text_color=self.SIDEBAR_COLORS['text_muted'], anchor="w"
        )
        text_label.pack(side="left", padx=(8, 4), fill="x", expand=True)
        self._collapsible_widgets.append(text_label)

        if shortcut:
            shortcut_label = ctk.CTkLabel(
                inner, text=shortcut, font=("Segoe UI", 9),
                text_color=self.SIDEBAR_COLORS['text_muted'], anchor="e"
            )
            shortcut_label.pack(side="right", padx=(0, 8))
            self._collapsible_widgets.append(shortcut_label)
            frame._shortcut_label = shortcut_label

        if tooltip:
            tooltip_text = f"{tooltip}\n\nRaccourci: {shortcut}" if shortcut else tooltip
            frame._tooltip = add_tooltip(frame, tooltip_text, position="right", delay=600)

        frame._indicator = indicator
        frame._icon_label = icon_label
        frame._text_label = text_label
        frame._is_active = False

        def on_click(e=None):
            self.select_button(index)
            command()

        def on_enter(e=None):
            if not frame._is_active:
                frame.configure(fg_color=self.SIDEBAR_COLORS['bg_secondary'])

        def on_leave(e=None):
            if not frame._is_active:
                frame.configure(fg_color="transparent")

        for w in [frame, inner, icon_label, text_label]:
            w.bind("<Button-1>", on_click)
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            try:
                w.configure(cursor="hand2")
            except Exception:
                pass

        def set_active(active):
            frame._is_active = active
            if active:
                frame.configure(
                    fg_color=self.SIDEBAR_COLORS['bg_active'],
                    border_width=1, border_color=self.SIDEBAR_COLORS['border_active']
                )
                indicator.configure(fg_color=self.SIDEBAR_COLORS['accent'])
                icon_label.configure(text_color=self.SIDEBAR_COLORS['accent'])
                text_label.configure(text_color=self.SIDEBAR_COLORS['text'])
            else:
                frame.configure(fg_color="transparent", border_width=0)
                indicator.configure(fg_color="transparent")
                icon_label.configure(text_color=self.SIDEBAR_COLORS['text_muted'])
                text_label.configure(text_color=self.SIDEBAR_COLORS['text_muted'])

        frame.set_active = set_active
        return frame

    def _bind_keyboard_shortcuts(self):
        """Lie les raccourcis clavier"""
        try:
            root = self.winfo_toplevel()
            mod = get_binding_modifier()

            for i, cmd in enumerate(self._nav_commands, start=1):
                if i > 8:
                    break
                def make_handler(index, command):
                    def handler(event=None):
                        self.select_button(index - 1)
                        command()
                        return "break"
                    return handler

                binding = f"<{mod}-Key-{i}>"
                root.bind(binding, make_handler(i, cmd))

        except Exception as e:
            logger.warning(f"Impossible de lier les raccourcis clavier: {e}")

    def toggle_sidebar(self):
        if self.is_collapsed:
            self._expand()
        else:
            self._collapse()

    def _collapse(self):
        self.is_collapsed = True
        for widget in self._collapsible_widgets:
            try:
                widget.pack_forget()
            except Exception:
                pass
        if hasattr(self, 'settings_btn'):
            self.settings_btn.pack_forget()
        self.configure(width=self.collapsed_width)
        self.toggle_btn.configure(text="☰")

    def _expand(self):
        self.is_collapsed = False
        self.configure(width=self.sidebar_width)
        self.logo_container.pack(side="left", padx=(12, 0), fill="y")
        for btn in self.buttons:
            if hasattr(btn, '_text_label'):
                btn._text_label.pack(side="left", padx=(8, 12), fill="x", expand=True)
        self.profile_label.pack(side="left", padx=(10, 0), fill="y")
        self.profile_chevron.pack(side="right", padx=(0, 10))
        self.settings_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.toggle_btn.configure(text="≡")

    def _do_logout(self):
        if hasattr(self, 'logout_callback') and self.logout_callback:
            self.logout_callback()
        else:
            try:
                self.winfo_toplevel().quit()
            except Exception:
                pass

    def _create_error_sidebar(self):
        error_frame = ctk.CTkFrame(self, fg_color='#21262d', corner_radius=10)
        error_frame.pack(fill="x", padx=16, pady=20)
        ctk.CTkLabel(
            error_frame, text=t('ui.error_navigation'),
            font=("Segoe UI", 12), text_color='#ff3860'
        ).pack(pady=16)

    def select_button(self, index):
        self._active_index = index
        for i, btn in enumerate(self.buttons):
            try:
                btn.set_active(i == index)
            except Exception:
                continue

    def reset_buttons(self):
        for btn in self.buttons:
            try:
                btn.set_active(False)
            except Exception:
                continue

    # === NAVIGATION METHODS ===

    def show_formation(self):
        """Affiche le module de formation"""
        try:
            safe_clear_main_frame()
            # Passer l'email utilisateur pour la progression personnalisée
            user_email = self.user_info.get('email', 'default')
            afficher_formation_commerciale(self.main_frame, user_email=user_email)
        except ImportError as e:
            logger.error(f"Module formation non trouvé: {e}")
            self.show_error_message("Module formation non disponible")
        except Exception as e:
            logger.error(f"Erreur chargement formation: {e}")
            self.show_error_message("Erreur de chargement de la formation")

    def show_profile(self):
        safe_afficher_profil()

    def show_settings(self):
        safe_afficher_parametres()

    def show_error_message(self, message):
        try:
            safe_clear_main_frame()
            error_frame = ctk.CTkFrame(self.main_frame, fg_color='#21262d', corner_radius=16)
            error_frame.pack(fill="both", expand=True, padx=40, pady=40)

            content = ctk.CTkFrame(error_frame, fg_color="transparent")
            content.place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(
                content, text=message, font=("Segoe UI", 16),
                text_color='#ff3860'
            ).pack(pady=(0, 24))

            ctk.CTkButton(
                content, text="Retour",
                command=lambda: self.select_button(0) or self.show_formation(),
                fg_color='#00BFFF', hover_color='#0090CC',
                corner_radius=10, height=40
            ).pack()
        except Exception as e:
            logger.error(f"Erreur affichage message d'erreur: {e}")


# === NOTIFICATIONS ===
class SafeToastNotification(ctk.CTkToplevel):
    """Notifications toast sécurisées"""

    def __init__(self, parent, message, type="info", duration=3000):
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
            except Exception:
                pass

    def _setup_position(self):
        try:
            self.update_idletasks()
            x = self.winfo_screenwidth() - 350
            y = self.winfo_screenheight() - 150
            self.geometry(f"320x80+{x}+{y}")
        except Exception:
            self.geometry("320x80+100+100")

    def _setup_content(self, message, type):
        colors = {
            "success": COLORS['accent_green'],
            "error": COLORS['accent_red'],
            "warning": COLORS['accent_yellow'],
            "info": COLORS['accent_blue']
        }
        icons = {"success": "✓", "error": "✗", "warning": "⚠", "info": "ℹ"}

        frame = ctk.CTkFrame(self, fg_color=COLORS['bg_hover'], corner_radius=10)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(
            frame, text=icons.get(type, "ℹ"), font=("Segoe UI", 20),
            text_color=colors.get(type, COLORS['accent_blue'])
        ).place(x=15, y=25)

        display_message = message[:100] + "..." if len(message) > 100 else message
        ctk.CTkLabel(
            frame, text=display_message, font=("Segoe UI", 12),
            text_color=COLORS['text_primary']
        ).place(x=50, y=28)

    def _schedule_close(self, duration):
        try:
            self.after(duration, self._safe_destroy)
        except Exception:
            pass

    def _animate_slide_in(self):
        try:
            self.attributes('-alpha', 0)
            for i in range(1, 11):
                self.after(i * 30, lambda a=i/10: self._set_alpha_safe(a))
        except Exception:
            pass

    def _set_alpha_safe(self, alpha):
        try:
            if self.winfo_exists():
                self.attributes('-alpha', alpha)
        except Exception:
            pass

    def _safe_destroy(self):
        try:
            if self.winfo_exists():
                self.destroy()
        except Exception:
            pass


# ============================================================================
# PANEL MANAGER
# ============================================================================
class PanelManager:
    def __init__(self, container=None):
        self.container = container
        self._panels = {}
        self._current_panel = None
        self._current_name = None
        self._transitioning = False

    def set_container(self, container):
        self.container = container

    def show_panel(self, name, create_func, transition='fade', force_recreate=False):
        if self._transitioning:
            return
        if not self.container:
            return
        if self._current_name == name and not force_recreate:
            return

        if name not in self._panels or force_recreate:
            if name in self._panels and force_recreate:
                try:
                    self._panels[name].destroy()
                except Exception:
                    pass
            try:
                panel = create_func(self.container)
                self._panels[name] = panel
            except Exception as e:
                logger.error(f"Erreur creation panel '{name}': {e}")
                return

        new_panel = self._panels[name]

        if self._current_panel:
            try:
                self._current_panel.pack_forget()
            except Exception:
                pass

        try:
            new_panel.pack(fill='both', expand=True)
        except Exception as e:
            logger.error(f"Erreur transition: {e}")

        self._current_panel = new_panel
        self._current_name = name

    def clear_cache(self, name=None):
        if name:
            if name in self._panels:
                try:
                    self._panels[name].destroy()
                except Exception:
                    pass
                del self._panels[name]
                if self._current_name == name:
                    self._current_panel = None
                    self._current_name = None
        else:
            for panel_name, panel in self._panels.items():
                try:
                    panel.destroy()
                except Exception:
                    pass
            self._panels.clear()
            self._current_panel = None
            self._current_name = None


panel_manager = PanelManager()


# ============================================================================
# VARIABLES GLOBALES
# ============================================================================
class AppState:
    def __init__(self):
        self.main_frame = None
        self.sidebar = None
        self.langue_var = None
        self.settings_vars = {}
        self.app = None
        self.container = None
        self.toast_manager = None
        self.panel_manager = panel_manager


app_state = AppState()

# Compatibilité
main_frame = None
sidebar = None
auth_manager_global = None
langue_var = None
toast_manager = None


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
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"Erreur nettoyage main_frame: {e}")


def safe_show_notification(message, type="info"):
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


def _apply_runtime_theme(theme_name):
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
            # Réafficher la formation
            safe_clear_main_frame()
            try:
                # Obtenir l'email utilisateur pour la progression personnalisée
                user_email = 'default'
                if auth_manager_global:
                    user = auth_manager_global.get_current_user()
                    user_email = user.get('email', 'default') if user else 'default'
                afficher_formation_commerciale(app_state.main_frame, user_email=user_email)
            except Exception:
                pass

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
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Erreur application thème: {e}")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def launch_main_app_with_auth(app, auth_manager, logout_callback):
    """Lance l'application principale avec authentification"""
    global main_frame, sidebar, langue_var, auth_manager_global

    try:
        auth_manager_global = auth_manager
        logger.info("Démarrage de l'application HelixOne avec authentification")

        # Mode DEV
        if os.environ.get("HELIXONE_DEV") == "1":
            from src.auth_session import set_auth_token
            dev_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjI2ZjI0MDctNGY2Yi00ODMyLWJjMTQtZGZhNzQ4M2JmY2Y0IiwiZW1haWwiOiJ0ZXN0QGhlbGl4b25lLmNvbSIsImV4cCI6MTc5MTkzMDA2N30.DDnZTWxmHCfPW6mVJrhKCU0HJeD7vCxcPTTIXwjmq5M"
            set_auth_token(dev_token)
            logger.info("MODE DEV: Token d'authentification défini")
        else:
            from src.auth_session import set_auth_token
            if hasattr(auth_manager, 'token') and auth_manager.token:
                set_auth_token(auth_manager.token)
                logger.info("Token d'authentification récupéré depuis auth_manager")

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

        # Sidebar
        try:
            sidebar = ModernSidebar(container, main_frame, user_info=user)
            sidebar.pack(side="left", fill="y")
            app_state.sidebar = sidebar
            sidebar.logout_callback = logout_callback
            logger.info("Sidebar créée avec succès")
        except Exception as e:
            logger.error(f"Erreur création sidebar: {e}")
            sidebar = create_fallback_sidebar(container)
            sidebar.pack(side="left", fill="y")

        main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        langue_var = ctk.StringVar(value=user_config["langue"])
        app_state.langue_var = langue_var

        # Raccourcis clavier
        mod = get_binding_modifier()
        app.bind(f'<{mod}-q>', lambda e: logout_callback())
        app.bind('<Escape>', lambda e: safe_clear_main_frame())

        # Afficher la Formation directement
        try:
            user_email = user.get('email', 'default') if user else 'default'
            afficher_formation_commerciale(main_frame, user_email=user_email)
        except Exception as e:
            logger.error(f"Erreur affichage formation: {e}")

        logger.info("Application HelixOne démarrée avec succès")

        # Notification de bienvenue
        def show_welcome():
            try:
                first_name = user.get('first_name', '') if user else ''
                WelcomeNotification(app, user_name=first_name)
            except Exception as welcome_err:
                logger.warning(f"Erreur notification bienvenue: {welcome_err}")

        app.after(2500, show_welcome)

    except Exception as e:
        logger.error(f"Erreur critique au démarrage: {e}")
        create_emergency_ui(app)


def launch_main_app(app, auth_manager=None, logout_callback=None):
    """Fonction d'entrée compatible avec et sans authentification"""
    if auth_manager and logout_callback:
        return launch_main_app_with_auth(app, auth_manager, logout_callback)
    else:
        return launch_main_app_old(app)


def launch_main_app_old(app):
    """Ancienne fonction pour compatibilité"""
    global main_frame, sidebar, langue_var, toast_manager

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

        toast_manager = ToastManager(app)
        app_state.toast_manager = toast_manager

        try:
            sidebar = ModernSidebar(container, main_frame)
            sidebar.pack(side="left", fill="y")
            app_state.sidebar = sidebar
        except Exception as e:
            logger.error(f"Erreur création sidebar: {e}")
            sidebar = create_fallback_sidebar(container)
            sidebar.pack(side="left", fill="y")

        main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        langue_var = ctk.StringVar(value=user_config["langue"])
        app_state.langue_var = langue_var

        mod = get_binding_modifier()
        app.bind(f'<{mod}-q>', lambda e: app.quit())
        app.bind('<Escape>', lambda e: safe_clear_main_frame())

        # Afficher la Formation directement
        try:
            user_email = user.get('email', 'default') if user else 'default'
            afficher_formation_commerciale(main_frame, user_email=user_email)
        except Exception as e:
            logger.error(f"Erreur affichage formation: {e}")

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

    def open_formation():
        user_email = 'default'
        if auth_manager_global:
            user = auth_manager_global.get_current_user()
            user_email = user.get('email', 'default') if user else 'default'
        afficher_formation_commerciale(main_frame, user_email=user_email)

    ctk.CTkButton(
        sidebar, text="Formation",
        command=open_formation
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
    except Exception:
        pass


# ============================================================================
# PAGES SECURISEES
# ============================================================================
def safe_afficher_parametres():
    """Page paramètres sécurisée"""
    try:
        safe_clear_main_frame()

        title = ctk.CTkLabel(
            main_frame, text="Paramètres", font=("Segoe UI", 28, "bold"),
            text_color=COLORS['text_primary']
        )
        title.pack(pady=(0, 20))

        settings_container = ctk.CTkScrollableFrame(
            main_frame, fg_color=COLORS['bg_secondary']
        )
        settings_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Section Apparence
        section_app = ctk.CTkFrame(settings_container, fg_color=COLORS['bg_tertiary'])
        section_app.pack(fill="x", pady=10, padx=20)

        ctk.CTkLabel(
            section_app, text="Apparence", font=("Segoe UI", 16, "bold"),
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

        # Bouton sauvegarder
        ctk.CTkButton(
            settings_container, text="Sauvegarder les paramètres",
            font=("Segoe UI", 14, "bold"), fg_color=COLORS['accent_green'],
            hover_color=COLORS['accent_blue'], command=save_settings
        ).pack(pady=30)

    except Exception as e:
        logger.error(f"Erreur page paramètres: {e}")
        show_error_page("Erreur Paramètres", str(e))


def safe_afficher_profil():
    """Page Profil Utilisateur"""
    try:
        safe_clear_main_frame()
        container = ctk.CTkFrame(main_frame, fg_color=COLORS['bg_primary'])
        container.pack(fill="both", expand=True)

        user_info = {}
        if auth_manager_global:
            try:
                user_info = auth_manager_global.get_current_user() or {}
            except Exception:
                pass

        profile_panel = ProfilePanel(
            container,
            user_info=user_info,
            auth_manager=auth_manager_global
        )
        profile_panel.pack(fill="both", expand=True)
    except Exception as e:
        logger.error(f"Erreur page Profil: {e}")
        show_error_page("Erreur Profil", str(e))


def save_settings():
    """Sauvegarde les paramètres"""
    try:
        if not app_state.settings_vars:
            safe_show_notification("Interface paramètres non initialisée", "error")
            return

        theme_ui = app_state.settings_vars["theme"].get()
        langue_ui = app_state.settings_vars["langue"].get()

        theme_cfg = theme_ui.lower()
        lang_cfg = "fr" if langue_ui.lower().startswith("fr") else "en"

        data = {}
        if config_manager._config_path.exists():
            try:
                with open(config_manager._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data.setdefault("app_settings", {})
        data["app_settings"]["theme"] = theme_cfg
        data["app_settings"]["language"] = lang_cfg

        with open(config_manager._config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if hasattr(config_manager.config, 'theme'):
            config_manager.config.theme = theme_cfg
        if hasattr(config_manager.config, 'language'):
            config_manager.config.language = lang_cfg

        user_config["theme"] = theme_cfg
        user_config["langue"] = "Français" if lang_cfg == "fr" else "English"

        _apply_runtime_theme(theme_cfg)

        try:
            set_language(lang_cfg)
            LanguagePreferences.save_language(lang_cfg)
        except Exception as e:
            logger.error(f"Erreur changement de langue: {e}")

        safe_show_notification(t('notifications.settings_saved'), "success")

    except Exception as e:
        logger.error(f"Erreur sauvegarde paramètres: {e}")
        safe_show_notification(t('notifications.save_error'), "error")


def show_error_page(title, message):
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

        def return_to_formation():
            user_email = 'default'
            if auth_manager_global:
                user = auth_manager_global.get_current_user()
                user_email = user.get('email', 'default') if user else 'default'
            afficher_formation_commerciale(main_frame, user_email=user_email)

        ctk.CTkButton(
            error_container, text="Retour à la Formation",
            command=return_to_formation,
            fg_color=COLORS['accent_blue']
        ).pack(pady=20)

    except Exception as e:
        logger.critical(f"Erreur critique affichage erreur: {e}")


# ============================================================================
# EXPORT
# ============================================================================
__all__ = [
    'launch_main_app',
    'safe_show_notification',
    'config_manager',
    'logger'
]
