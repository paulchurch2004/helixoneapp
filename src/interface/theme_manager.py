"""
Gestionnaire de thèmes premium pour l'interface
Thèmes : Dark, Light, Cyberpunk, Forest, Ocean
"""

import customtkinter as ctk
from typing import Dict
import json
import os
from pathlib import Path

class ThemeManager:
    """Gestionnaire centralisé des thèmes"""
    
    THEMES = {
        'dark': {
            'name': 'Dark Professional',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#0a0e27',
                'bg_secondary': '#1a1e3f',
                'bg_tertiary': '#2a2e4f',
                'text_primary': '#ffffff',
                'text_secondary': '#94a3b8',
                'accent_primary': '#4a9eff',
                'accent_secondary': '#00d4ff',
                'success': '#00ff88',
                'warning': '#ffaa00',
                'error': '#ff4444',
                'border': '#334155',
                'shadow': '#00000099'
            },
            'effects': {
                'particle_color': '#4a9eff',
                'glow_color': '#00d4ff',
                'gradient_start': '#1e3a8a',
                'gradient_end': '#3b82f6'
            }
        },
        
        'light': {
            'name': 'Light Modern',
            'mode': 'light',
            'colors': {
                'bg_primary': '#f8fafc',
                'bg_secondary': '#f1f5f9',
                'bg_tertiary': '#e2e8f0',
                'text_primary': '#0f172a',
                'text_secondary': '#475569',
                'accent_primary': '#2563eb',
                'accent_secondary': '#3b82f6',
                'success': '#10b981',
                'warning': '#f59e0b',
                'error': '#ef4444',
                'border': '#cbd5e1',
                'shadow': '#00000033'
            },
            'effects': {
                'particle_color': '#2563eb',
                'glow_color': '#3b82f6',
                'gradient_start': '#dbeafe',
                'gradient_end': '#bfdbfe'
            }
        },
        
        'cyberpunk': {
            'name': 'Cyberpunk Neon',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#0d0221',
                'bg_secondary': '#1a0b2e',
                'bg_tertiary': '#2d1b4e',
                'text_primary': '#00f5ff',
                'text_secondary': '#ff006e',
                'accent_primary': '#ff006e',
                'accent_secondary': '#8338ec',
                'success': '#00ff88',
                'warning': '#ffbe0b',
                'error': '#ff006e',
                'border': '#8338ec',
                'shadow': '#ff006e66'
            },
            'effects': {
                'particle_color': '#ff006e',
                'glow_color': '#00f5ff',
                'gradient_start': '#8338ec',
                'gradient_end': '#ff006e'
            }
        },
        
        'forest': {
            'name': 'Forest Green',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#0f1419',
                'bg_secondary': '#1a2229',
                'bg_tertiary': '#253039',
                'text_primary': '#e0f2e9',
                'text_secondary': '#a0c4b8',
                'accent_primary': '#00ff88',
                'accent_secondary': '#00aa55',
                'success': '#00ff88',
                'warning': '#ffaa00',
                'error': '#ff5544',
                'border': '#2d5540',
                'shadow': '#00000099'
            },
            'effects': {
                'particle_color': '#00ff88',
                'glow_color': '#00ffaa',
                'gradient_start': '#1a4d2e',
                'gradient_end': '#00ff88'
            }
        },
        
        'ocean': {
            'name': 'Ocean Blue',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#001f3f',
                'bg_secondary': '#003366',
                'bg_tertiary': '#004d7a',
                'text_primary': '#e0f2fe',
                'text_secondary': '#a5d8ff',
                'accent_primary': '#00b4d8',
                'accent_secondary': '#0096c7',
                'success': '#06ffa5',
                'warning': '#ffd60a',
                'error': '#ff477e',
                'border': '#005f8f',
                'shadow': '#00000099'
            },
            'effects': {
                'particle_color': '#00b4d8',
                'glow_color': '#00d4ff',
                'gradient_start': '#003566',
                'gradient_end': '#00b4d8'
            }
        }
    }
    
    def __init__(self):
        self.current_theme = 'dark'
        self.config_file = Path('data/theme_config.json')
        self._load_config()
    
    def _load_config(self):
        """Charge la configuration du thème sauvegardé"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.current_theme = config.get('theme', 'dark')
        except Exception as e:
            print(f"Erreur chargement config thème: {e}")
    
    def _save_config(self):
        """Sauvegarde la configuration du thème"""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump({'theme': self.current_theme}, f)
        except Exception as e:
            print(f"Erreur sauvegarde config thème: {e}")
    
    def get_current_theme(self) -> Dict:
        """Retourne le thème actuel"""
        return self.THEMES.get(self.current_theme, self.THEMES['dark'])
    
    def set_theme(self, theme_name: str):
        """Change le thème actuel"""
        if theme_name in self.THEMES:
            self.current_theme = theme_name
            self._save_config()
            self._apply_theme()
    
    def _apply_theme(self):
        """Applique le thème à customtkinter"""
        theme = self.get_current_theme()
        ctk.set_appearance_mode(theme['mode'])
    
    def get_color(self, color_key: str) -> str:
        """Récupère une couleur du thème actuel"""
        theme = self.get_current_theme()
        return theme['colors'].get(color_key, '#ffffff')
    
    def get_effect_color(self, effect_key: str) -> str:
        """Récupère une couleur d'effet du thème actuel"""
        theme = self.get_current_theme()
        return theme['effects'].get(effect_key, '#4a9eff')
    
    def get_all_themes(self) -> Dict:
        """Retourne tous les thèmes disponibles"""
        return self.THEMES
    
    def create_theme_selector(self, parent) -> ctk.CTkFrame:
        """Crée un sélecteur de thème visuel"""
        frame = ctk.CTkFrame(parent)
        
        title = ctk.CTkLabel(
            frame,
            text="Thème",
            font=('Helvetica', 16, 'bold')
        )
        title.pack(pady=(10, 20))
        
        for theme_key, theme_data in self.THEMES.items():
            btn = ctk.CTkButton(
                frame,
                text=theme_data['name'],
                command=lambda t=theme_key: self._on_theme_select(t),
                fg_color=theme_data['colors']['accent_primary'],
                hover_color=theme_data['colors']['accent_secondary'],
                width=200
            )
            btn.pack(pady=5)
            
            if theme_key == self.current_theme:
                btn.configure(text=f"✓ {theme_data['name']}")
        
        return frame
    
    def _on_theme_select(self, theme_name: str):
        """Callback lors de la sélection d'un thème"""
        self.set_theme(theme_name)
        print(f"Thème changé: {self.THEMES[theme_name]['name']}")


class AnimatedThemeTransition:
    """Transition animée entre les thèmes"""
    
    @staticmethod
    def fade_transition(widget, duration: int = 500):
        """Transition par fondu"""
        steps = 20
        delay = duration // steps
        
        def fade_step(step: int = 0):
            if step >= steps:
                return
            
            widget.after(delay, lambda: fade_step(step + 1))
        
        fade_step()
    
    @staticmethod
    def slide_transition(widget, direction: str = 'left', duration: int = 300):
        """Transition par glissement"""
        start_x = widget.winfo_x()
        start_y = widget.winfo_y()
        
        distance = widget.winfo_width() if direction in ['left', 'right'] else widget.winfo_height()
        
        if direction in ['left', 'up']:
            distance = -distance
        
        steps = 20
        delay = duration // steps
        step_size = distance / steps
        
        def slide_step(step: int = 0):
            if step >= steps:
                widget.place(x=start_x, y=start_y)
                return
            
            if direction in ['left', 'right']:
                widget.place(x=start_x + step_size * step, y=start_y)
            else:
                widget.place(x=start_x, y=start_y + step_size * step)
            
            widget.after(delay, lambda: slide_step(step + 1))

        slide_step()


# Instance globale du gestionnaire de thèmes
theme_manager = ThemeManager()