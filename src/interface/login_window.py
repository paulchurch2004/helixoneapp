"""
Fenêtre de connexion HelixOne
"""

import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
from pathlib import Path
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.helixone_client import HelixOneAPIError
from src.i18n import t
from src.asset_path import get_asset_path


class LoginWindow(ctk.CTk):
    """
    Fenêtre de connexion
    """
    
    def __init__(self, auth_manager):
        super().__init__()

        self.auth_manager = auth_manager
        self.register_callback = None
        self.success_callback = None

        # Configuration fenêtre
        self.title(f"HelixOne - {t('auth.login')}")
        self.geometry("450x650")  # Augmenté pour bouton biométrique
        self.resizable(False, False)

        # Centrer la fenêtre
        self.center_window()

        # Vérifier si connexion rapide disponible
        self.quick_login_available = self.auth_manager.is_quick_login_enabled()
        self.biometric_available = self.auth_manager.is_biometric_available()

        # Créer l'interface
        self.create_ui()

        # Si connexion rapide activée ET biométrie dispo, proposer auto-login
        if self.quick_login_available and self.biometric_available:
            self.after(500, self._prompt_biometric_login)
    
    def center_window(self):
        """Centrer la fenêtre sur l'écran"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_ui(self):
        """Créer l'interface utilisateur"""
        
        # Container principal
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Logo image
        try:
            logo_path = get_asset_path("logo.png")
            if logo_path:
                logo_image = Image.open(logo_path)
                # Redimensionner pour la page login (max 150px de large)
                ratio = logo_image.width / logo_image.height
                new_width = min(150, logo_image.width)
                new_height = int(new_width / ratio)

                self.logo_ctk = ctk.CTkImage(
                    light_image=logo_image,
                    dark_image=logo_image,
                    size=(new_width, new_height)
                )

                logo_label = ctk.CTkLabel(
                    main_frame,
                    image=self.logo_ctk,
                    text=""
                )
                logo_label.pack(pady=(0, 10))
            else:
                raise FileNotFoundError()
        except Exception:
            # Fallback texte si logo non trouvé
            title_label = ctk.CTkLabel(
                main_frame,
                text="HelixOne",
                font=("Arial", 36, "bold"),
                text_color="#1f538d"
            )
            title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text=t('app.subtitle'),
            font=("Arial", 14),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 20))

        # Bouton connexion biométrique (si disponible)
        if self.quick_login_available and self.biometric_available:
            biometry_type = self.auth_manager.get_biometry_type()
            quick_email = self.auth_manager.get_quick_login_email()

            if biometry_type == "touchid":
                icon = "👆"
                text = f"Connexion rapide avec Touch ID"
            elif biometry_type == "faceid":
                icon = "😀"
                text = f"Connexion rapide avec Face ID"
            else:
                icon = "🔐"
                text = "Connexion rapide"

            biometric_button = ctk.CTkButton(
                main_frame,
                text=f"{icon} {text}",
                height=50,
                font=("Arial", 14, "bold"),
                fg_color="#4CAF50",
                hover_color="#45a049",
                command=self._handle_biometric_login
            )
            biometric_button.pack(fill="x", pady=(0, 10))

            # Label email pour connexion rapide
            quick_email_label = ctk.CTkLabel(
                main_frame,
                text=f"Connecté en tant que {quick_email}",
                font=("Arial", 10),
                text_color="gray"
            )
            quick_email_label.pack(pady=(0, 20))

            # Séparateur
            separator = ctk.CTkLabel(
                main_frame,
                text="─── ou ───",
                font=("Arial", 11),
                text_color="gray"
            )
            separator.pack(pady=(0, 20))

        # Email
        email_label = ctk.CTkLabel(
            main_frame,
            text=t('auth.email'),
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        email_label.pack(fill="x", pady=(0, 5))

        self.email_entry = ctk.CTkEntry(
            main_frame,
            height=40,
            placeholder_text=t('auth.email_placeholder'),
            font=("Arial", 12)
        )
        self.email_entry.pack(fill="x", pady=(0, 20))

        # Mot de passe
        password_label = ctk.CTkLabel(
            main_frame,
            text=t('auth.password'),
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        password_label.pack(fill="x", pady=(0, 5))

        self.password_entry = ctk.CTkEntry(
            main_frame,
            height=40,
            show="•",
            placeholder_text=t('auth.password_placeholder'),
            font=("Arial", 12)
        )
        self.password_entry.pack(fill="x", pady=(0, 10))

        # Bind Enter key
        self.password_entry.bind("<Return>", lambda e: self.handle_login())

        # Checkbox "Se souvenir de cet appareil"
        remember_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        remember_frame.pack(fill="x", pady=(0, 10))

        self.remember_device_var = ctk.BooleanVar(
            value=self.auth_manager.is_quick_login_enabled()
        )

        self.remember_checkbox = ctk.CTkCheckBox(
            remember_frame,
            text="Se souvenir de cet appareil",
            variable=self.remember_device_var,
            font=("Arial", 11),
            command=self._on_remember_changed
        )
        self.remember_checkbox.pack(side="left")

        # Info biométrie si disponible
        if self.biometric_available and not self.quick_login_available:
            biometry_type = self.auth_manager.get_biometry_type()
            if biometry_type == "touchid":
                bio_text = "(Touch ID)"
            elif biometry_type == "faceid":
                bio_text = "(Face ID)"
            else:
                bio_text = ""

            if bio_text:
                bio_label = ctk.CTkLabel(
                    remember_frame,
                    text=bio_text,
                    font=("Arial", 10),
                    text_color="#4CAF50"
                )
                bio_label.pack(side="left", padx=(5, 0))

        # Message d'erreur
        self.error_label = ctk.CTkLabel(
            main_frame,
            text="",
            text_color="red",
            font=("Arial", 11)
        )
        self.error_label.pack(pady=(0, 10))
        
        # Bouton connexion
        login_button = ctk.CTkButton(
            main_frame,
            text=t('auth.login_button'),
            height=40,
            font=("Arial", 14, "bold"),
            command=self.handle_login
        )
        login_button.pack(fill="x", pady=(10, 20))

        # Séparateur
        separator_frame = ctk.CTkFrame(main_frame, height=1, fg_color="gray")
        separator_frame.pack(fill="x", pady=20)

        # Texte "Pas de compte ?"
        no_account_label = ctk.CTkLabel(
            main_frame,
            text=t('auth.register'),
            font=("Arial", 11),
            text_color="gray"
        )
        no_account_label.pack()

        # Bouton inscription
        register_button = ctk.CTkButton(
            main_frame,
            text=t('auth.register_button'),
            height=40,
            font=("Arial", 13),
            fg_color="transparent",
            border_width=2,
            border_color="#1f538d",
            hover_color="#e8f0f8",
            text_color="#1f538d",
            command=self.show_register
        )
        register_button.pack(fill="x", pady=(10, 0))
    
    def handle_login(self, totp_code: str = None):
        """Gérer la connexion (avec support 2FA)"""
        email = self.email_entry.get().strip()
        password = self.password_entry.get()

        # Validation
        if not email:
            self.error_label.configure(text=f"⚠️ {t('auth.email')}")
            return

        if not password:
            self.error_label.configure(text=f"⚠️ {t('auth.password')}")
            return

        # Désactiver le bouton pendant la requête
        self.error_label.configure(text=f"⏳ {t('app.loading')}", text_color="blue")
        self.update()

        try:
            # Appeler l'API avec le code 2FA si fourni
            result = self.auth_manager.login(email, password, totp_code)

            # Vérifier si 2FA est requis
            if result.get("requires_2fa"):
                self.error_label.configure(text="", text_color="red")
                self._show_2fa_dialog(email, password)
                return

            # Si "Se souvenir de cet appareil" est coché, activer connexion rapide
            if self.remember_device_var.get():
                self.auth_manager.enable_quick_login(email, password)

            # Récupérer la licence
            license = self.auth_manager.get_license_info()

            # Afficher message de succès
            messagebox.showinfo(
                t('auth.login_success'),
                f"{t('app.success')} !\n\n"
                f"Licence : {license['license_type'].upper()}\n"
                f"Jours restants : {license['days_remaining']}"
            )

            # Callback de succès
            if self.success_callback:
                self.success_callback()

            # Fermer la fenêtre
            self.destroy()

        except HelixOneAPIError as e:
            error_msg = str(e)
            if "401" in error_msg or "incorrect" in error_msg.lower():
                self.error_label.configure(
                    text=f"❌ {t('auth.invalid_credentials')}",
                    text_color="red"
                )
            else:
                self.error_label.configure(
                    text=f"❌ {error_msg}",
                    text_color="red"
                )
        except Exception as e:
            self.error_label.configure(
                text=f"❌ {t('app.error')}: {str(e)}",
                text_color="red"
            )

    def _handle_biometric_login(self):
        """Gérer la connexion avec biométrie"""
        self.error_label.configure(text="🔐 Authentification...", text_color="blue")
        self.update()

        def on_result(success, error):
            if success:
                # Connexion réussie
                try:
                    license = self.auth_manager.get_license_info()
                    messagebox.showinfo(
                        t('auth.login_success'),
                        f"{t('app.success')} !\n\n"
                        f"Licence : {license['license_type'].upper()}\n"
                        f"Jours restants : {license['days_remaining']}"
                    )

                    if self.success_callback:
                        self.success_callback()

                    self.destroy()
                except Exception as e:
                    self.error_label.configure(
                        text=f"❌ Erreur: {str(e)}",
                        text_color="red"
                    )
            else:
                # Authentification échouée
                self.error_label.configure(
                    text=f"❌ {error or 'Authentification annulée'}",
                    text_color="red"
                )

        self.auth_manager.biometric_login(callback=on_result)

    def _prompt_biometric_login(self):
        """Proposer la connexion biométrique au démarrage"""
        # Auto-déclencher la connexion biométrique
        self._handle_biometric_login()

    def _on_remember_changed(self):
        """Callback quand la checkbox 'Se souvenir' change"""
        if not self.remember_device_var.get():
            # Si décochée, désactiver la connexion rapide
            if self.auth_manager.is_quick_login_enabled():
                self.auth_manager.disable_quick_login()
    
    def _show_2fa_dialog(self, email: str, password: str):
        """Affiche le dialogue pour entrer le code 2FA"""
        dialog = TwoFALoginDialog(self, email, password, self.auth_manager, self._on_2fa_success)

    def _on_2fa_success(self):
        """Callback après 2FA réussi"""
        # Récupérer la licence
        try:
            license = self.auth_manager.get_license_info()
            messagebox.showinfo(
                t('auth.login_success'),
                f"{t('app.success')} !\n\n"
                f"Licence : {license['license_type'].upper()}\n"
                f"Jours restants : {license['days_remaining']}"
            )
        except Exception:
            pass

        if self.success_callback:
            self.success_callback()
        self.destroy()

    def show_register(self):
        if self.register_callback:
            self.register_callback()

    def set_register_callback(self, callback):
        """Définir la fonction à appeler pour l'inscription"""
        self.register_callback = callback

    def set_success_callback(self, callback):
        """Définir la fonction à appeler après connexion réussie"""
        self.success_callback = callback


class TwoFALoginDialog(ctk.CTkToplevel):
    """Dialogue pour entrer le code 2FA lors de la connexion"""

    def __init__(self, parent, email: str, password: str, auth_manager, success_callback):
        super().__init__(parent)

        self.email = email
        self.password = password
        self.auth_manager = auth_manager
        self.success_callback = success_callback

        self.title("Authentification 2FA")
        self.geometry("400x300")
        self.configure(fg_color="#0f1117")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._create_ui()

        # Centrer
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 300) // 2
        self.geometry(f"+{x}+{y}")

    def _create_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1d24", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🔐 Vérification 2FA",
            font=("Arial", 18, "bold"),
            text_color="#FFFFFF"
        ).pack(side="left", padx=25, pady=15)

        # Content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=25)

        ctk.CTkLabel(
            content,
            text="Entrez le code à 6 chiffres de votre\napp d'authentification",
            font=("Arial", 14),
            text_color="#AAAAAA",
            justify="center"
        ).pack(pady=(0, 20))

        # Code entry
        self.code_entry = ctk.CTkEntry(
            content,
            height=60,
            font=("Courier", 28, "bold"),
            fg_color="#1a1d24",
            border_color="#2a2d36",
            justify="center",
            placeholder_text="000000"
        )
        self.code_entry.pack(fill="x", pady=(0, 15))
        self.code_entry.bind("<Return>", lambda e: self._verify())
        self.code_entry.focus()

        # Error label
        self.error_label = ctk.CTkLabel(
            content,
            text="",
            font=("Arial", 12),
            text_color="#FF4444"
        )
        self.error_label.pack(pady=(0, 10))

        # Buttons
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame,
            text="Annuler",
            font=("Arial", 13),
            fg_color="#2a2d36",
            hover_color="#3a3d46",
            height=42,
            corner_radius=8,
            command=self.destroy
        ).pack(side="left", expand=True, padx=(0, 5))

        self.verify_btn = ctk.CTkButton(
            btn_frame,
            text="Vérifier",
            font=("Arial", 13, "bold"),
            fg_color="#00D9FF",
            hover_color="#00B8E6",
            text_color="#000000",
            height=42,
            corner_radius=8,
            command=self._verify
        )
        self.verify_btn.pack(side="right", expand=True, padx=(5, 0))

    def _verify(self):
        """Vérifie le code 2FA"""
        code = self.code_entry.get().strip()

        if not code or len(code) != 6 or not code.isdigit():
            self.error_label.configure(text="Entrez un code à 6 chiffres")
            return

        try:
            self.verify_btn.configure(state="disabled", text="Vérification...")
            self.update()

            # Tenter la connexion avec le code 2FA
            result = self.auth_manager.login(self.email, self.password, code)

            if result.get("access_token"):
                # Succès
                self.destroy()
                self.success_callback()
            else:
                self.error_label.configure(text="Erreur de connexion")
                self.verify_btn.configure(state="normal", text="Vérifier")

        except Exception as e:
            error_msg = str(e)
            if "invalide" in error_msg.lower() or "2fa" in error_msg.lower():
                self.error_label.configure(text="Code incorrect")
            else:
                self.error_label.configure(text=f"Erreur: {error_msg}")
            self.verify_btn.configure(state="normal", text="Vérifier")


# Test du module
if __name__ == "__main__":
    from src.auth_manager import AuthManager
    
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    auth = AuthManager()
    
    def on_success():
        print("✅ Connexion réussie !")
    
    def on_register():
        print("📝 Afficher l'inscription")
    
    app = LoginWindow(auth)
    app.set_success_callback(on_success)
    app.set_register_callback(on_register)
    app.mainloop()
