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
from src.interface.onboarding_wizard import reset_onboarding, has_completed_onboarding


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

        # === SECTION NOTIFICATIONS ===
        self.create_notifications_section()

        # Séparateur
        separator3 = ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray30")
        separator3.pack(fill="x", pady=20)

        # === SECTION ACCESSIBILITE ===
        self.create_accessibility_section()

        # Séparateur
        separator4 = ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray30")
        separator4.pack(fill="x", pady=20)

        # === SECTION AIDE ===
        self.create_help_section()

        # Séparateur
        separator5 = ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray30")
        separator5.pack(fill="x", pady=20)

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
            "fr": "🇫🇷 Français",
            "en": "🇬🇧 English"
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

    def create_notifications_section(self):
        """Créer la section notifications"""

        notif_frame = ctk.CTkFrame(self.scroll_frame, fg_color="gray20", corner_radius=10)
        notif_frame.pack(fill="x", pady=10)

        # Titre
        notif_title = ctk.CTkLabel(
            notif_frame,
            text=t('settings.notifications'),
            font=("Roboto", 18, "bold"),
            anchor="w"
        )
        notif_title.pack(fill="x", padx=20, pady=(20, 10))

        # Switch notifications
        self.notifications_switch = ctk.CTkSwitch(
            notif_frame,
            text=t('alerts.title'),
            font=("Roboto", 14),
            fg_color="#1f538d",
            progress_color="#2a6cb8"
        )
        self.notifications_switch.select()
        self.notifications_switch.pack(anchor="w", padx=20, pady=(0, 20))

    def create_accessibility_section(self):
        """Creer la section accessibilite"""

        access_frame = ctk.CTkFrame(self.scroll_frame, fg_color="gray20", corner_radius=10)
        access_frame.pack(fill="x", pady=10)

        # Titre
        access_title = ctk.CTkLabel(
            access_frame,
            text="Accessibilité",
            font=("Roboto", 18, "bold"),
            anchor="w"
        )
        access_title.pack(fill="x", padx=20, pady=(20, 10))

        # Description
        ctk.CTkLabel(
            access_frame,
            text="Options pour ameliorer l'experience utilisateur",
            font=("Roboto", 12),
            text_color="gray70",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 15))

        # --- Taille de police ---
        font_frame = ctk.CTkFrame(access_frame, fg_color="transparent")
        font_frame.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(
            font_frame,
            text="Taille de police:",
            font=("Roboto", 14),
            anchor="w"
        ).pack(side="left")

        self.font_size_var = ctk.StringVar(value="Normal")
        font_options = ctk.CTkSegmentedButton(
            font_frame,
            values=["Petit", "Normal", "Grand", "Tres grand"],
            variable=self.font_size_var,
            command=self._on_font_size_change,
            font=("Roboto", 12)
        )
        font_options.pack(side="right", padx=(10, 0))

        # --- Mode contraste eleve ---
        self.high_contrast_switch = ctk.CTkSwitch(
            access_frame,
            text="Mode contraste eleve",
            font=("Roboto", 14),
            fg_color="#1f538d",
            progress_color="#2a6cb8",
            command=self._on_high_contrast_change
        )
        self.high_contrast_switch.pack(anchor="w", padx=20, pady=(0, 10))

        # --- Reduire les animations ---
        self.reduce_motion_switch = ctk.CTkSwitch(
            access_frame,
            text="Reduire les animations",
            font=("Roboto", 14),
            fg_color="#1f538d",
            progress_color="#2a6cb8",
            command=self._on_reduce_motion_change
        )
        self.reduce_motion_switch.pack(anchor="w", padx=20, pady=(0, 20))

    def create_help_section(self):
        """Creer la section aide et tutoriels"""

        help_frame = ctk.CTkFrame(self.scroll_frame, fg_color="gray20", corner_radius=10)
        help_frame.pack(fill="x", pady=10)

        # Titre
        help_title = ctk.CTkLabel(
            help_frame,
            text="Aide & Tutoriels",
            font=("Roboto", 18, "bold"),
            anchor="w"
        )
        help_title.pack(fill="x", padx=20, pady=(20, 10))

        # Description
        ctk.CTkLabel(
            help_frame,
            text="Ressources pour apprendre a utiliser HelixOne",
            font=("Roboto", 12),
            text_color="gray70",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 15))

        # Bouton raccourcis clavier
        shortcuts_btn = ctk.CTkButton(
            help_frame,
            text="Voir les raccourcis clavier (F1)",
            font=("Roboto", 14),
            fg_color="gray30",
            hover_color="gray40",
            anchor="w",
            command=self._show_shortcuts
        )
        shortcuts_btn.pack(fill="x", padx=20, pady=(0, 10))

        # Bouton relancer l'introduction
        onboarding_btn = ctk.CTkButton(
            help_frame,
            text="Relancer l'introduction",
            font=("Roboto", 14),
            fg_color="gray30",
            hover_color="gray40",
            anchor="w",
            command=self._restart_onboarding
        )
        onboarding_btn.pack(fill="x", padx=20, pady=(0, 10))

        # Bouton formation
        formation_btn = ctk.CTkButton(
            help_frame,
            text="Accéder à la Formation",
            font=("Roboto", 14),
            fg_color="gray30",
            hover_color="gray40",
            anchor="w",
            command=self._go_to_formation
        )
        formation_btn.pack(fill="x", padx=20, pady=(0, 20))

    def _on_font_size_change(self, value: str):
        """Change la taille de police"""
        size_map = {
            "Petit": 12,
            "Normal": 14,
            "Grand": 16,
            "Tres grand": 18
        }
        # Stocker pour application ulterieure
        print(f"[Settings] Taille police: {value} ({size_map.get(value, 14)}px)")
        # TODO: Appliquer globalement

    def _on_high_contrast_change(self):
        """Active/desactive le mode contraste eleve"""
        is_enabled = self.high_contrast_switch.get()
        print(f"[Settings] Contraste eleve: {is_enabled}")
        # TODO: Appliquer le theme contraste eleve

    def _on_reduce_motion_change(self):
        """Active/desactive la reduction des animations"""
        is_enabled = self.reduce_motion_switch.get()
        print(f"[Settings] Reduire animations: {is_enabled}")
        # TODO: Desactiver les animations

    def _show_shortcuts(self):
        """Affiche la fenetre des raccourcis clavier"""
        try:
            # Simuler F1
            if self.app_instance:
                event = type('Event', (), {'keysym': 'F1'})()
                self.app_instance.event_generate('<F1>')
        except Exception as e:
            print(f"[Settings] Erreur affichage raccourcis: {e}")
            messagebox.showinfo(
                "Raccourcis clavier",
                "Appuyez sur F1 a tout moment pour voir les raccourcis."
            )

    def _restart_onboarding(self):
        """Relance le tutoriel d'introduction"""
        try:
            reset_onboarding()
            messagebox.showinfo(
                "Introduction",
                "L'introduction sera affichee au prochain demarrage.\n\n"
                "Vous pouvez aussi fermer et rouvrir HelixOne maintenant."
            )
        except Exception as e:
            print(f"[Settings] Erreur reset onboarding: {e}")

    def _go_to_formation(self):
        """Navigue vers la section Formation"""
        try:
            if self.app_instance and hasattr(self.app_instance, 'sidebar'):
                # Trouver et cliquer sur le bouton Formation
                self.app_instance.sidebar.afficher_panel("formation")
        except Exception as e:
            print(f"[Settings] Erreur navigation formation: {e}")

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
