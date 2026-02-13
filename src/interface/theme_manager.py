"""
Gestionnaire de thèmes premium pour l'interface
Thèmes : Dark, Light, Cyberpunk, Forest, Ocean
Avec support callbacks et persistence améliorée
"""

import customtkinter as ctk
from typing import Dict, Callable, List
import json
import os
from pathlib import Path

class ThemeManager:
    """Gestionnaire centralisé des thèmes avec callbacks"""

    THEMES = {
        'dark': {
            'name': 'Dark Professional',
            'icon': '🌙',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#0a0e27',
                'bg_secondary': '#1a1e3f',
                'bg_tertiary': '#2a2e4f',
                'bg_card': '#1e2346',
                'bg_hover': '#2e3256',
                'text_primary': '#ffffff',
                'text_secondary': '#94a3b8',
                'text_muted': '#64748b',
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
            'icon': '☀️',
            'mode': 'light',
            'colors': {
                'bg_primary': '#f8fafc',
                'bg_secondary': '#f1f5f9',
                'bg_tertiary': '#e2e8f0',
                'bg_card': '#ffffff',
                'bg_hover': '#e8ecf2',
                'text_primary': '#0f172a',
                'text_secondary': '#475569',
                'text_muted': '#94a3b8',
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
            'icon': '🎮',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#0d0221',
                'bg_secondary': '#1a0b2e',
                'bg_tertiary': '#2d1b4e',
                'bg_card': '#1f0d38',
                'bg_hover': '#3d2b5e',
                'text_primary': '#00f5ff',
                'text_secondary': '#ff006e',
                'text_muted': '#8866aa',
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
            'icon': '🌲',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#0f1419',
                'bg_secondary': '#1a2229',
                'bg_tertiary': '#253039',
                'bg_card': '#1d2a2f',
                'bg_hover': '#2f3f49',
                'text_primary': '#e0f2e9',
                'text_secondary': '#a0c4b8',
                'text_muted': '#6a9080',
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
            'icon': '🌊',
            'mode': 'dark',
            'colors': {
                'bg_primary': '#001f3f',
                'bg_secondary': '#003366',
                'bg_tertiary': '#004d7a',
                'bg_card': '#00446f',
                'bg_hover': '#005d8f',
                'text_primary': '#e0f2fe',
                'text_secondary': '#a5d8ff',
                'text_muted': '#74c0fc',
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
        self.config_dir = Path.home() / '.helixone'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / 'theme_config.json'
        self.callbacks: List[Callable[[Dict], None]] = []
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
            with open(self.config_file, 'w') as f:
                json.dump({'theme': self.current_theme}, f)
        except Exception as e:
            print(f"Erreur sauvegarde config thème: {e}")

    def get_current_theme(self) -> Dict:
        """Retourne le thème actuel"""
        return self.THEMES.get(self.current_theme, self.THEMES['dark'])

    def get_theme_name(self) -> str:
        """Retourne le nom du thème actuel"""
        return self.current_theme

    def set_theme(self, theme_name: str, show_restart_message: bool = True):
        """Change le thème actuel"""
        if theme_name in self.THEMES:
            old_theme = self.current_theme
            self.current_theme = theme_name
            self._save_config()
            self._apply_theme()
            self._notify_callbacks()

            # Afficher message si le thème a vraiment changé
            if show_restart_message and old_theme != theme_name:
                self._show_restart_message()

    def toggle_dark_light(self):
        """Bascule entre dark et light"""
        if self.current_theme == 'dark':
            self.set_theme('light')
        else:
            self.set_theme('dark')

    def _apply_theme(self):
        """Applique le thème à customtkinter"""
        theme = self.get_current_theme()
        ctk.set_appearance_mode(theme['mode'])

    def _show_restart_message(self):
        """Affiche un message demandant de redémarrer l'application"""
        try:
            from tkinter import messagebox
            messagebox.showinfo(
                "Changement de thème",
                "Le thème a été changé.\n\n"
                "Pour appliquer complètement les nouvelles couleurs,\n"
                "veuillez redémarrer l'application.",
                icon='info'
            )
        except Exception as e:
            print(f"Erreur affichage message restart: {e}")

    def register_callback(self, callback: Callable[[Dict], None]):
        """Enregistre un callback pour les changements de thème"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Dict], None]):
        """Supprime un callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _notify_callbacks(self):
        """Notifie tous les callbacks du changement"""
        theme = self.get_current_theme()
        for callback in self.callbacks:
            try:
                callback(theme)
            except Exception as e:
                print(f"Erreur callback thème: {e}")

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


def get_theme_manager() -> ThemeManager:
    """Retourne l'instance globale du gestionnaire de thèmes"""
    return theme_manager


class ThemeSwitcher(ctk.CTkFrame):
    """Widget de sélection de thème compact ou complet"""

    def __init__(self, parent, compact: bool = False, **kwargs):
        super().__init__(parent, **kwargs)
        self.manager = get_theme_manager()
        self.compact = compact
        theme = self.manager.get_current_theme()
        self.configure(fg_color="transparent")
        self._create_ui()

    def _create_ui(self):
        """Crée l'interface"""
        if self.compact:
            self._create_compact_ui()
        else:
            self._create_full_ui()

    def _create_compact_ui(self):
        """Interface compacte (bouton unique pour toggle)"""
        theme = self.manager.get_current_theme()
        icon = theme.get('icon', '🌙')

        self.toggle_btn = ctk.CTkButton(
            self,
            text=icon,
            width=40,
            height=40,
            fg_color=theme['colors']['bg_tertiary'],
            hover_color=theme['colors']['bg_hover'],
            command=self._toggle_theme,
            font=ctk.CTkFont(size=18)
        )
        self.toggle_btn.pack()

    def _create_full_ui(self):
        """Interface complète avec tous les thèmes"""
        theme = self.manager.get_current_theme()

        # Titre
        ctk.CTkLabel(
            self,
            text="🎨 Thème",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=theme['colors']['text_primary']
        ).pack(anchor="w", pady=(0, 10))

        # Grille de thèmes
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(fill="x")

        themes = list(self.manager.THEMES.items())
        for i, (theme_key, theme_data) in enumerate(themes):
            is_current = theme_key == self.manager.current_theme

            card = ctk.CTkFrame(
                grid,
                fg_color=theme_data['colors']['bg_secondary'],
                border_width=3 if is_current else 1,
                border_color=theme_data['colors']['accent_primary'] if is_current else theme_data['colors']['border'],
                corner_radius=10,
                cursor="hand2"
            )
            card.grid(row=i//3, column=i%3, padx=5, pady=5, sticky="nsew")
            grid.columnconfigure(i%3, weight=1)

            # Preview colors
            preview = ctk.CTkFrame(card, fg_color=theme_data['colors']['bg_primary'], corner_radius=5, height=30)
            preview.pack(fill="x", padx=8, pady=(8, 4))
            preview.pack_propagate(False)

            ctk.CTkFrame(preview, fg_color=theme_data['colors']['accent_primary'], width=15, height=6, corner_radius=2).place(x=5, y=5)
            ctk.CTkFrame(preview, fg_color=theme_data['colors']['text_secondary'], width=30, height=4, corner_radius=2).place(x=25, y=6)
            ctk.CTkFrame(preview, fg_color=theme_data['colors']['success'], width=8, height=8, corner_radius=4).place(x=5, y=16)

            # Label
            ctk.CTkLabel(
                card,
                text=f"{theme_data['icon']} {theme_data['name'].split()[0]}",
                font=ctk.CTkFont(size=11),
                text_color=theme_data['colors']['text_primary']
            ).pack(pady=(4, 8))

            # Bind click
            card.bind("<Button-1>", lambda e, tk=theme_key: self._select_theme(tk))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, tk=theme_key: self._select_theme(tk))

    def _toggle_theme(self):
        """Bascule entre dark et light"""
        self.manager.toggle_dark_light()
        self._update_ui()

    def _select_theme(self, theme_key: str):
        """Sélectionne un thème"""
        self.manager.set_theme(theme_key)
        self._update_ui()

    def _update_ui(self):
        """Met à jour l'interface"""
        for widget in self.winfo_children():
            widget.destroy()
        self._create_ui()


class ThemeSettingsPanel(ctk.CTkFrame):
    """Panneau de paramètres de thème complet"""

    def __init__(self, parent, **kwargs):
        self.manager = get_theme_manager()
        theme = self.manager.get_current_theme()
        super().__init__(parent, fg_color=theme['colors']['bg_secondary'], corner_radius=15, **kwargs)
        self._create_ui()
        self.manager.register_callback(self._on_theme_change)

    def _create_ui(self):
        """Crée l'interface du panneau"""
        theme = self.manager.get_current_theme()

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            header,
            text="🎨 Personnalisation",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=theme['colors']['text_primary']
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Choisissez l'apparence de HelixOne",
            font=ctk.CTkFont(size=13),
            text_color=theme['colors']['text_muted']
        ).pack(anchor="w", pady=(5, 0))

        # Switcher
        switcher = ThemeSwitcher(self, compact=False)
        switcher.pack(fill="x", padx=20, pady=10)

        # Quick toggle
        toggle_frame = ctk.CTkFrame(self, fg_color=theme['colors']['bg_tertiary'], corner_radius=10)
        toggle_frame.pack(fill="x", padx=20, pady=15)

        ctk.CTkLabel(
            toggle_frame,
            text="Raccourci: Ctrl+T pour changer de thème",
            font=ctk.CTkFont(size=12),
            text_color=theme['colors']['text_secondary']
        ).pack(pady=15)

        # Preview
        self._create_preview()

    def _create_preview(self):
        """Crée un aperçu du thème"""
        theme = self.manager.get_current_theme()

        preview_container = ctk.CTkFrame(self, fg_color=theme['colors']['bg_tertiary'], corner_radius=10)
        preview_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            preview_container,
            text="👁️ Aperçu",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=theme['colors']['text_primary']
        ).pack(anchor="w", padx=15, pady=(15, 10))

        preview = ctk.CTkFrame(preview_container, fg_color=theme['colors']['bg_primary'], corner_radius=8)
        preview.pack(fill="x", padx=15, pady=(0, 15))

        # Mini header
        mini_header = ctk.CTkFrame(preview, fg_color=theme['colors']['bg_secondary'], height=25, corner_radius=0)
        mini_header.pack(fill="x")
        mini_header.pack_propagate(False)

        ctk.CTkLabel(
            mini_header,
            text="  HelixOne",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=theme['colors']['accent_primary']
        ).pack(side="left", pady=3)

        # Content
        content = ctk.CTkFrame(preview, fg_color="transparent")
        content.pack(fill="x", padx=8, pady=8)

        # Cards
        for ticker, change, positive in [("AAPL", "+2.45%", True), ("TSLA", "-1.23%", False)]:
            card = ctk.CTkFrame(content, fg_color=theme['colors']['bg_card'], corner_radius=5)
            card.pack(fill="x", pady=2)

            ctk.CTkLabel(
                card,
                text=ticker,
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color=theme['colors']['text_primary']
            ).pack(side="left", padx=8, pady=6)

            color = theme['colors']['success'] if positive else theme['colors']['error']
            ctk.CTkLabel(
                card,
                text=change,
                font=ctk.CTkFont(size=9),
                text_color=color
            ).pack(side="right", padx=8, pady=6)

        # Button
        ctk.CTkButton(
            content,
            text="Analyser",
            font=ctk.CTkFont(size=9),
            height=22,
            fg_color=theme['colors']['accent_primary']
        ).pack(fill="x", pady=(8, 4))

    def _on_theme_change(self, theme: Dict):
        """Appelé lors du changement de thème"""
        self.configure(fg_color=theme['colors']['bg_secondary'])
        for widget in self.winfo_children():
            widget.destroy()
        self._create_ui()

    def destroy(self):
        """Nettoyage"""
        self.manager.unregister_callback(self._on_theme_change)
        super().destroy()