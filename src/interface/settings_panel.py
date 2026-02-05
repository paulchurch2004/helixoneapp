"""
Panneau des paramètres avec support multilingue
"""

import customtkinter as ctk
from tkinter import messagebox
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.i18n import t, set_language, get_language, get_available_languages, LanguagePreferences
from src.interface.theme_manager import get_theme_manager, ThemeSwitcher


class SettingsPanel(ctk.CTkFrame):
    """Panneau de paramètres avec changement de langue"""

    def __init__(self, master, app_instance=None):
        super().__init__(master, fg_color="transparent")

        self.app_instance = app_instance
        self.language_callbacks = []  # Callbacks à appeler lors du changement de langue

        self.create_ui()

    def create_ui(self):
        """Créer l'interface des paramètres"""

        # Container principal avec scroll
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        title_label = ctk.CTkLabel(
            self.scroll_frame,
            text=t('settings.title'),
            font=("Roboto", 24, "bold")
        )
        title_label.pack(pady=(0, 30))

        # === SECTION LANGUE ===
        self.create_language_section()

        # Séparateur
        separator1 = ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray30")
        separator1.pack(fill="x", pady=20)

        # === SECTION THÈME ===
        self.create_theme_section()

        # Séparateur
        separator2 = ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray30")
        separator2.pack(fill="x", pady=20)

        # === BOUTON SAUVEGARDER ===
        save_button = ctk.CTkButton(
            self.scroll_frame,
            text=t('settings.save_settings'),
            command=self.save_settings,
            height=40,
            font=("Roboto", 14, "bold"),
            fg_color="#1f538d",
            hover_color="#2a6cb8"
        )
        save_button.pack(pady=20)

    def create_language_section(self):
        """Créer la section de sélection de langue"""

        # Frame de la section
        lang_frame = ctk.CTkFrame(self.scroll_frame, fg_color="gray20", corner_radius=10)
        lang_frame.pack(fill="x", pady=10)

        # Titre de section
        lang_title = ctk.CTkLabel(
            lang_frame,
            text=t('settings.language'),
            font=("Roboto", 18, "bold"),
            anchor="w"
        )
        lang_title.pack(fill="x", padx=20, pady=(20, 10))

        # Description
        lang_desc = ctk.CTkLabel(
            lang_frame,
            text=t('settings.language_selector'),
            font=("Roboto", 12),
            text_color="gray70",
            anchor="w"
        )
        lang_desc.pack(fill="x", padx=20, pady=(0, 15))

        # Frame pour les boutons radio de langue
        radio_frame = ctk.CTkFrame(lang_frame, fg_color="transparent")
        radio_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Variable pour stocker la langue sélectionnée
        self.language_var = ctk.StringVar(value=get_language())

        # Créer un bouton radio pour chaque langue disponible
        languages = {
            "fr": "Français",
            "en": "English"
        }

        for lang_code, lang_name in languages.items():
            radio_btn = ctk.CTkRadioButton(
                radio_frame,
                text=lang_name,
                variable=self.language_var,
                value=lang_code,
                command=lambda lc=lang_code: self.on_language_change(lc),
                font=("Roboto", 14),
                fg_color="#1f538d",
                hover_color="#2a6cb8"
            )
            radio_btn.pack(anchor="w", pady=5, padx=10)

    def create_theme_section(self):
        """Créer la section de thème"""

        theme_frame = ctk.CTkFrame(self.scroll_frame, fg_color="gray20", corner_radius=10)
        theme_frame.pack(fill="x", pady=10)

        # Titre
        theme_title = ctk.CTkLabel(
            theme_frame,
            text=t('settings.theme'),
            font=("Roboto", 18, "bold"),
            anchor="w"
        )
        theme_title.pack(fill="x", padx=20, pady=(20, 10))

        # Description
        ctk.CTkLabel(
            theme_frame,
            text="Choisissez l'apparence de HelixOne",
            font=("Roboto", 12),
            text_color="gray70",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 10))

        # Sélecteur de thème complet
        try:
            theme_switcher = ThemeSwitcher(theme_frame, compact=False)
            theme_switcher.pack(fill="x", padx=20, pady=(0, 20))
        except Exception as e:
            # Fallback: switch simple
            print(f"Erreur ThemeSwitcher: {e}")
            self.dark_mode_switch = ctk.CTkSwitch(
                theme_frame,
                text=t('settings.dark_mode'),
                font=("Roboto", 14),
                fg_color="#1f538d",
                progress_color="#2a6cb8",
                command=self._toggle_theme
            )
            self.dark_mode_switch.select()
            self.dark_mode_switch.pack(anchor="w", padx=20, pady=(0, 20))

    def _toggle_theme(self):
        """Bascule le thème clair/sombre"""
        try:
            manager = get_theme_manager()
            manager.toggle_dark_light()
        except Exception as e:
            print(f"Erreur changement thème: {e}")

    def on_language_change(self, lang_code: str):
        """
        Callback appelé lors du changement de langue

        Args:
            lang_code: Code de la langue (fr, en)
        """
        if set_language(lang_code):
            # Sauvegarder la préférence
            LanguagePreferences.save_language(lang_code)

            # Afficher un message de confirmation
            messagebox.showinfo(
                t('app.success'),
                t('settings.settings_saved')
            )

            # Recharger l'interface si on a une référence à l'app
            if self.app_instance and hasattr(self.app_instance, 'reload_interface'):
                self.app_instance.reload_interface()

            # Appeler les callbacks enregistrés
            for callback in self.language_callbacks:
                try:
                    callback(lang_code)
                except Exception as e:
                    print(f"[Settings] Erreur callback langue: {e}")

            # Rafraîchir l'interface actuelle
            self.refresh_texts()

    def refresh_texts(self):
        """Rafraîchit tous les textes de l'interface avec la nouvelle langue"""
        try:
            # Cette méthode sera appelée pour mettre à jour les textes
            # après un changement de langue
            self.destroy_ui()
            self.create_ui()
        except Exception as e:
            print(f"[Settings] Erreur rafraichissement interface: {e}")

    def destroy_ui(self):
        """Détruit l'interface actuelle"""
        for widget in self.winfo_children():
            widget.destroy()

    def register_language_callback(self, callback):
        """
        Enregistre un callback à appeler lors du changement de langue

        Args:
            callback: Fonction à appeler avec le code de langue en paramètre
        """
        if callback not in self.language_callbacks:
            self.language_callbacks.append(callback)

    def save_settings(self):
        """Sauvegarder tous les paramètres"""
        try:
            # Sauvegarder la langue
            current_lang = self.language_var.get()
            LanguagePreferences.save_language(current_lang)

            # Afficher confirmation
            messagebox.showinfo(
                t('app.success'),
                t('settings.settings_saved')
            )
        except Exception as e:
            messagebox.showerror(
                t('app.error'),
                f"{t('errors.save_failed')}: {str(e)}"
            )
