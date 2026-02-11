import customtkinter as ctk
from PIL import Image
import yfinance as yf
import threading
import time
import os
from tkinter import messagebox
import requests
import sys
from src.asset_path import get_asset_path

# Import du module de gestion de session
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.auth_session import set_auth_token
from src.config import get_api_url

INDICES = {
    "^FCHI": "CAC 40",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW JONES",
    "^RUT": "RUSSELL 2000"
}

# Cache pour les données (éviter trop de requêtes)
INDICES_CACHE = {}
CACHE_DURATION = 300  # 5 minutes


class HomePanel(ctk.CTkFrame):
    def __init__(self, master, on_continue_callback):
        super().__init__(master, fg_color="#0a0a0f", corner_radius=0)
        self.indice_labels = {}
        self.on_continue_callback = on_continue_callback
        self._destroyed = False
        self._update_thread = None
        self._stop_updates = False
        self._server_ready = False
        self._status_label = None

        self.place_ui()
        self.after(1000, self.afficher_logo)
        self.after(1500, self.afficher_formulaire_connexion)
        self.after(2000, self.afficher_indices)

        # Wake-up ping au serveur en arrière-plan
        threading.Thread(target=self._wake_up_server, daemon=True).start()

    def _wake_up_server(self):
        """Envoie un ping au serveur pour le réveiller (cold start Render)"""
        try:
            requests.get(f"{get_api_url()}/", timeout=120)
            self._server_ready = True
            print("[OK] Serveur réveillé")
        except Exception:
            self._server_ready = False
            print("[WARN] Serveur non joignable")

    def _show_status(self, text):
        """Affiche un message de statut sous le formulaire"""
        if self._destroyed:
            return
        if self._status_label:
            self._status_label.destroy()
        self._status_label = ctk.CTkLabel(
            self, text=text,
            font=("Segoe UI", 12),
            text_color="#00BFFF"
        )
        self._status_label.place(relx=0.5, rely=0.92, anchor="center")

    def _hide_status(self):
        """Cache le message de statut"""
        if self._status_label:
            self._status_label.destroy()
            self._status_label = None

    def place_ui(self):
        # Fond uni qui couvre toute la fenêtre (évite les bandes noires)
        self._bg_frame = ctk.CTkFrame(self, fg_color="#0a0a0f", corner_radius=0)
        self._bg_frame.place(x=0, y=0, relwidth=1, relheight=1)

        # Image de fond par-dessus (optionnelle)
        try:
            self._bg_original = Image.open(get_asset_path("fond_texture.png"))
            self._bg_label = ctk.CTkLabel(self._bg_frame, text="", fg_color="transparent")
            self._bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            # Mettre à jour l'image quand la fenêtre change de taille
            self.bind("<Configure>", self._on_resize)
            self.after(100, self._update_background)
        except Exception as e:
            print(f"[⚠️] Erreur fond : {e}")

        # Barre supérieure pour les indices (créée une seule fois)
        self.top_frame = ctk.CTkFrame(self, fg_color="#1c1f26", height=60, corner_radius=10)
        self.top_frame.place(relx=0.5, rely=0.05, anchor="n", relwidth=0.92)

    def _on_resize(self, event=None):
        """Redimensionne le fond quand la fenêtre change de taille"""
        # Mise à jour immédiate pour éviter les bandes noires
        self._update_background()

    def _update_background(self):
        """Met à jour l'image de fond pour couvrir toute la fenêtre"""
        if self._destroyed or not hasattr(self, '_bg_original'):
            return
        try:
            w = self.winfo_width()
            h = self.winfo_height()
            if w > 1 and h > 1:
                # Redimensionner l'image pour couvrir la fenêtre
                bg_resized = self._bg_original.resize((w, h), Image.Resampling.LANCZOS)
                bg_ctk = ctk.CTkImage(light_image=bg_resized, dark_image=bg_resized, size=(w, h))
                self._bg_label.configure(image=bg_ctk)
                self._bg_label.image = bg_ctk
                # S'assurer que les widgets sont au-dessus du fond
                if hasattr(self, 'top_frame'):
                    self.top_frame.lift()
        except Exception as e:
            print(f"[⚠️] Erreur resize fond : {e}")

    def afficher_logo(self):
        if self._destroyed:
            return
        try:
            logo_img = Image.open(get_asset_path("logo.png")).resize((160, 160))
            logo = ctk.CTkImage(light_image=logo_img, size=(160, 160))
            logo_label = ctk.CTkLabel(self, image=logo, text="")
            logo_label.image = logo
            logo_label.place(relx=0.5, rely=0.3, anchor="center")
        except Exception as e:
            print(f"[⚠️] Erreur logo : {e}")

    def afficher_formulaire_connexion(self):
        if self._destroyed:
            return

        self.email_entry = ctk.CTkEntry(self, placeholder_text="Adresse Email", width=300)
        self.email_entry.place(relx=0.5, rely=0.52, anchor="center")

        self.password_entry = ctk.CTkEntry(self, placeholder_text="Mot de passe", show="*", width=300)
        self.password_entry.place(relx=0.5, rely=0.59, anchor="center")

        btn_connexion = ctk.CTkButton(
            self,
            text="Connexion",
            fg_color="#00BFFF",
            hover_color="#0090FF",
            corner_radius=15,
            font=("Segoe UI", 16, "bold"),
            command=self.verifier_connexion
        )
        btn_connexion.place(relx=0.5, rely=0.67, anchor="center")

        # Lien "Mot de passe oublié"
        forgot_password_label = ctk.CTkLabel(
            self,
            text="Mot de passe oublié ?",
            font=("Segoe UI", 11, "underline"),
            text_color="#00BFFF",
            cursor="hand2"
        )
        forgot_password_label.place(relx=0.5, rely=0.72, anchor="center")
        forgot_password_label.bind("<Button-1>", lambda e: self.afficher_formulaire_reset_password())

        # Séparateur "ou"
        separator_label = ctk.CTkLabel(
            self,
            text="─────────  ou  ─────────",
            font=("Segoe UI", 12),
            text_color="#666666"
        )
        separator_label.place(relx=0.5, rely=0.77, anchor="center")

        # Bouton Créer un compte
        btn_register = ctk.CTkButton(
            self,
            text="Créer un compte",
            fg_color="transparent",
            hover_color="#1c1f26",
            border_width=2,
            border_color="#00BFFF",
            text_color="#00BFFF",
            corner_radius=15,
            font=("Segoe UI", 14),
            command=self.afficher_formulaire_inscription
        )
        btn_register.place(relx=0.5, rely=0.84, anchor="center")

    def verifier_connexion(self):
        if self._destroyed:
            return

        email = self.email_entry.get()
        password = self.password_entry.get()

        if not email or not password:
            messagebox.showerror("Erreur", "Email et mot de passe requis.")
            return

        self._show_status("Connexion en cours... (le serveur peut mettre quelques secondes)")
        threading.Thread(target=self._do_login, args=(email, password), daemon=True).start()

    def _do_login(self, email, password):
        success, error_msg = self.authentifier_utilisateur(email, password)
        if self._destroyed:
            return
        self.after(0, lambda: self._on_login_result(success, error_msg))

    def _on_login_result(self, success, error_msg):
        self._hide_status()
        if self._destroyed:
            return
        if success:
            self._destroyed = True
            self._stop_updates = True
            self.on_continue_callback()
        else:
            messagebox.showerror("Erreur", error_msg)

    def _evaluer_force_mdp(self, password):
        """Évalue la force d'un mot de passe. Retourne (score, label, couleur)."""
        if not password:
            return (0, "", "#333333")

        score = 0
        if len(password) >= 8:
            score += 25
        if len(password) >= 12:
            score += 15
        if len(password) >= 16:
            score += 10
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" for c in password):
            score += 20

        score = min(score, 100)

        if score <= 25:
            return (score, "Faible", "#FF3333")
        elif score <= 50:
            return (score, "Moyen", "#FF8800")
        elif score <= 75:
            return (score, "Fort", "#BBCC00")
        else:
            return (score, "Très fort", "#00CC44")

    def _on_password_change(self, event=None):
        """Met à jour la barre de force quand le mot de passe change."""
        if self._destroyed:
            return
        password = self.register_password_entry.get()
        score, label, color = self._evaluer_force_mdp(password)

        try:
            if password:
                self._strength_bar.configure(fg_color=color)
                self._strength_bar.place(relx=0, rely=0, relheight=1, relwidth=max(score / 100, 0.02))
                self._strength_label.configure(text=label, text_color=color)
            else:
                self._strength_bar.place(relx=0, rely=0, relheight=1, relwidth=0)
                self._strength_label.configure(text="", text_color="#666666")
        except Exception:
            pass

    def afficher_formulaire_inscription(self):
        """Affiche le formulaire d'inscription"""
        if self._destroyed:
            return

        # Cacher TOUS les widgets existants (y compris frames)
        for widget in self.winfo_children():
            widget.place_forget()

        # Titre
        title_label = ctk.CTkLabel(
            self,
            text="Créer un compte",
            font=("Segoe UI", 22, "bold"),
            text_color="#00BFFF"
        )
        title_label.place(relx=0.5, rely=0.05, anchor="center")
        self._register_widgets = [title_label]

        # Prénom
        self.firstname_entry = ctk.CTkEntry(self, placeholder_text="Prénom", width=300, height=32)
        self.firstname_entry.place(relx=0.5, rely=0.14, anchor="center")
        self._register_widgets.append(self.firstname_entry)

        # Nom
        self.lastname_entry = ctk.CTkEntry(self, placeholder_text="Nom", width=300, height=32)
        self.lastname_entry.place(relx=0.5, rely=0.22, anchor="center")
        self._register_widgets.append(self.lastname_entry)

        # Email
        self.register_email_entry = ctk.CTkEntry(self, placeholder_text="Adresse Email", width=300, height=32)
        self.register_email_entry.place(relx=0.5, rely=0.30, anchor="center")
        self._register_widgets.append(self.register_email_entry)

        # Mot de passe
        self.register_password_entry = ctk.CTkEntry(self, placeholder_text="Mot de passe (12+ caractères)", show="*", width=300, height=32)
        self.register_password_entry.place(relx=0.5, rely=0.38, anchor="center")
        self.register_password_entry.bind("<KeyRelease>", self._on_password_change)
        self._register_widgets.append(self.register_password_entry)

        # Barre de force du mot de passe
        self._strength_bg = ctk.CTkFrame(self, width=300, height=5, fg_color="#333333", corner_radius=3)
        self._strength_bg.place(relx=0.5, rely=0.425, anchor="center")
        self._strength_bar = ctk.CTkFrame(self._strength_bg, height=5, fg_color="#333333", corner_radius=3)
        self._strength_bar.place(relx=0, rely=0, relheight=1, relwidth=0)
        self._register_widgets.append(self._strength_bg)

        self._strength_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 9), text_color="#666666"
        )
        self._strength_label.place(relx=0.5, rely=0.45, anchor="center")
        self._register_widgets.append(self._strength_label)

        # Confirmer le mot de passe
        self.register_confirm_entry = ctk.CTkEntry(self, placeholder_text="Confirmer le mot de passe", show="*", width=300, height=32)
        self.register_confirm_entry.place(relx=0.5, rely=0.50, anchor="center")
        self._register_widgets.append(self.register_confirm_entry)

        # Info mot de passe
        info_label = ctk.CTkLabel(
            self,
            text="Majuscule, minuscule, chiffre et caractère spécial requis",
            font=("Segoe UI", 9),
            text_color="#666666"
        )
        info_label.place(relx=0.5, rely=0.56, anchor="center")
        self._register_widgets.append(info_label)

        # Bouton inscription
        btn_signup = ctk.CTkButton(
            self,
            text="S'inscrire",
            fg_color="#00BFFF",
            hover_color="#0090FF",
            corner_radius=15,
            height=40,
            font=("Segoe UI", 16, "bold"),
            command=self.creer_compte
        )
        btn_signup.place(relx=0.5, rely=0.64, anchor="center")
        self._register_widgets.append(btn_signup)

        # Bouton retour
        btn_back = ctk.CTkButton(
            self,
            text="← Retour à la connexion",
            fg_color="transparent",
            hover_color="#1c1f26",
            text_color="#00BFFF",
            corner_radius=15,
            font=("Segoe UI", 12),
            command=self.retour_connexion
        )
        btn_back.place(relx=0.5, rely=0.72, anchor="center")
        self._register_widgets.append(btn_back)

    def creer_compte(self):
        """Crée un nouveau compte utilisateur"""
        if self._destroyed:
            return

        first_name = self.firstname_entry.get().strip()
        last_name = self.lastname_entry.get().strip()
        email = self.register_email_entry.get().strip()
        password = self.register_password_entry.get()
        confirm_password = self.register_confirm_entry.get()

        # Validation basique
        if not email or not password:
            messagebox.showerror("Erreur", "Email et mot de passe requis.")
            return

        if password != confirm_password:
            messagebox.showerror("Erreur", "Les mots de passe ne correspondent pas.")
            return

        if len(password) < 12:
            messagebox.showerror("Erreur", "Le mot de passe doit contenir au moins 12 caractères.")
            return

        self._show_status("Création du compte en cours...")
        threading.Thread(target=self._do_register, args=(email, password, first_name, last_name), daemon=True).start()

    def _do_register(self, email, password, first_name, last_name):
        try:
            response = requests.post(
                f"{get_api_url()}/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "first_name": first_name if first_name else None,
                    "last_name": last_name if last_name else None
                },
                timeout=120
            )
            self.after(0, lambda: self._on_register_result(response))
        except Exception as e:
            self.after(0, lambda: self._on_register_error(str(e)))

    def _on_register_result(self, response):
        self._hide_status()
        if self._destroyed:
            return
        if response.status_code == 201:
            messagebox.showinfo(
                "Succès",
                "Compte créé avec succès !\n\nVous pouvez maintenant vous connecter."
            )
            self.retour_connexion()
        else:
            error_detail = response.json().get("detail", "Erreur inconnue")
            messagebox.showerror("Erreur", f"Impossible de créer le compte:\n{error_detail}")

    def _on_register_error(self, error):
        self._hide_status()
        if self._destroyed:
            return
        messagebox.showerror("Erreur", f"Le serveur met du temps à répondre.\nRéessayez dans quelques secondes.\n\n{error}")

    def afficher_formulaire_reset_password(self):
        """Affiche le formulaire de réinitialisation du mot de passe"""
        if self._destroyed:
            return

        # Cacher les widgets de connexion
        for widget in self.winfo_children():
            if isinstance(widget, (ctk.CTkEntry, ctk.CTkButton, ctk.CTkLabel)):
                if widget != self.top_frame:
                    widget.place_forget()

        # Titre
        title_label = ctk.CTkLabel(
            self,
            text="Mot de passe oublié",
            font=("Segoe UI", 24, "bold"),
            text_color="#00BFFF"
        )
        title_label.place(relx=0.5, rely=0.40, anchor="center")

        # Info
        info_label = ctk.CTkLabel(
            self,
            text="Entrez votre adresse email pour recevoir\nun code de réinitialisation",
            font=("Segoe UI", 12),
            text_color="#CCCCCC"
        )
        info_label.place(relx=0.5, rely=0.48, anchor="center")

        # Email
        self.reset_email_entry = ctk.CTkEntry(self, placeholder_text="Adresse Email", width=300)
        self.reset_email_entry.place(relx=0.5, rely=0.56, anchor="center")

        # Bouton envoyer
        btn_send = ctk.CTkButton(
            self,
            text="Envoyer le code",
            fg_color="#00BFFF",
            hover_color="#0090FF",
            corner_radius=15,
            font=("Segoe UI", 16, "bold"),
            command=self.envoyer_code_reset
        )
        btn_send.place(relx=0.5, rely=0.66, anchor="center")

        # Bouton retour
        btn_back = ctk.CTkButton(
            self,
            text="← Retour à la connexion",
            fg_color="transparent",
            hover_color="#1c1f26",
            text_color="#00BFFF",
            corner_radius=15,
            font=("Segoe UI", 12),
            command=self.retour_connexion
        )
        btn_back.place(relx=0.5, rely=0.75, anchor="center")

    def envoyer_code_reset(self):
        """Envoie un code de réinitialisation"""
        if self._destroyed:
            return

        email = self.reset_email_entry.get().strip()

        if not email:
            messagebox.showerror("Erreur", "Veuillez entrer votre adresse email.")
            return

        self._show_status("Envoi du code en cours...")
        threading.Thread(target=self._do_reset, args=(email,), daemon=True).start()

    def _do_reset(self, email):
        try:
            response = requests.post(
                f"{get_api_url()}/auth/forgot-password",
                json={"email": email},
                timeout=120
            )
            self.after(0, lambda: self._on_reset_result(response))
        except Exception as e:
            self.after(0, lambda: self._on_reset_error(str(e)))

    def _on_reset_result(self, response):
        self._hide_status()
        if self._destroyed:
            return
        if response.status_code == 200:
            messagebox.showinfo(
                "Email envoyé",
                "Un code de réinitialisation a été envoyé à votre adresse email.\n\n"
                "Vérifiez votre boîte de réception (et vos spams)."
            )
            self.afficher_formulaire_nouveau_mdp()
        else:
            error_detail = response.json().get("detail", "Erreur inconnue")
            messagebox.showerror("Erreur", f"Impossible d'envoyer le code:\n{error_detail}")

    def _on_reset_error(self, error):
        self._hide_status()
        if self._destroyed:
            return
        messagebox.showerror("Erreur", f"Le serveur met du temps à répondre.\nRéessayez dans quelques secondes.\n\n{error}")

    def afficher_formulaire_nouveau_mdp(self):
        """Affiche le formulaire pour entrer le nouveau mot de passe"""
        if self._destroyed:
            return

        # Cacher les widgets
        for widget in self.winfo_children():
            if isinstance(widget, (ctk.CTkEntry, ctk.CTkButton, ctk.CTkLabel)):
                if widget != self.top_frame:
                    widget.place_forget()

        # Titre
        title_label = ctk.CTkLabel(
            self,
            text="Nouveau mot de passe",
            font=("Segoe UI", 24, "bold"),
            text_color="#00BFFF"
        )
        title_label.place(relx=0.5, rely=0.35, anchor="center")

        # Email (lecture seule)
        email_display = ctk.CTkLabel(
            self,
            text=f"Email: {self.reset_email_entry.get()}",
            font=("Segoe UI", 11),
            text_color="#CCCCCC"
        )
        email_display.place(relx=0.5, rely=0.43, anchor="center")

        # Code de réinitialisation
        self.reset_code_entry = ctk.CTkEntry(self, placeholder_text="Code de réinitialisation", width=300)
        self.reset_code_entry.place(relx=0.5, rely=0.50, anchor="center")

        # Nouveau mot de passe
        self.new_password_entry = ctk.CTkEntry(self, placeholder_text="Nouveau mot de passe", show="*", width=300)
        self.new_password_entry.place(relx=0.5, rely=0.57, anchor="center")

        # Confirmation
        self.confirm_password_entry = ctk.CTkEntry(self, placeholder_text="Confirmer le mot de passe", show="*", width=300)
        self.confirm_password_entry.place(relx=0.5, rely=0.64, anchor="center")

        # Info
        info_label = ctk.CTkLabel(
            self,
            text="12+ caractères, majuscule, minuscule, chiffre, spécial",
            font=("Segoe UI", 9),
            text_color="#666666"
        )
        info_label.place(relx=0.5, rely=0.69, anchor="center")

        # Bouton valider
        btn_validate = ctk.CTkButton(
            self,
            text="Réinitialiser le mot de passe",
            fg_color="#00BFFF",
            hover_color="#0090FF",
            corner_radius=15,
            font=("Segoe UI", 14, "bold"),
            command=self.reinitialiser_mot_de_passe
        )
        btn_validate.place(relx=0.5, rely=0.77, anchor="center")

        # Bouton retour
        btn_back = ctk.CTkButton(
            self,
            text="← Retour",
            fg_color="transparent",
            hover_color="#1c1f26",
            text_color="#00BFFF",
            corner_radius=15,
            font=("Segoe UI", 12),
            command=self.retour_connexion
        )
        btn_back.place(relx=0.5, rely=0.85, anchor="center")

    def reinitialiser_mot_de_passe(self):
        """Réinitialise le mot de passe avec le code"""
        if self._destroyed:
            return

        email = self.reset_email_entry.get().strip()
        code = self.reset_code_entry.get().strip()
        new_password = self.new_password_entry.get()
        confirm_password = self.confirm_password_entry.get()

        # Validations
        if not code or not new_password:
            messagebox.showerror("Erreur", "Tous les champs sont requis.")
            return

        if new_password != confirm_password:
            messagebox.showerror("Erreur", "Les mots de passe ne correspondent pas.")
            return

        if len(new_password) < 12:
            messagebox.showerror("Erreur", "Le mot de passe doit contenir au moins 12 caractères.")
            return

        try:
            response = requests.post(
                f"{get_api_url()}/auth/reset-password",
                json={
                    "email": email,
                    "reset_code": code,
                    "new_password": new_password
                },
                timeout=120
            )

            if response.status_code == 200:
                messagebox.showinfo(
                    "Succès",
                    "Mot de passe réinitialisé avec succès !\n\nVous pouvez maintenant vous connecter."
                )
                self.retour_connexion()
            else:
                error_detail = response.json().get("detail", "Erreur inconnue")
                messagebox.showerror("Erreur", f"Impossible de réinitialiser:\n{error_detail}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Le serveur met du temps à répondre.\nRéessayez dans quelques secondes.\n\n{str(e)}")

    def retour_connexion(self):
        """Retourne au formulaire de connexion"""
        if self._destroyed:
            return

        # Supprimer tous les widgets sauf le top_frame
        for widget in self.winfo_children():
            if widget != self.top_frame:
                widget.destroy()

        # Réafficher le formulaire de connexion
        self.after(100, self.afficher_logo)
        self.after(200, self.afficher_formulaire_connexion)

    def authentifier_utilisateur(self, email, password):
        # === MODE DEV : AUTO-CONNEXION ===
        if os.environ.get("HELIXONE_DEV") == "1":
            print("[⚙️ MODE DEV] Login bypassé automatiquement.")
            set_auth_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjI2ZjI0MDctNGY2Yi00ODMyLWJjMTQtZGZhNzQ4M2JmY2Y0IiwiZW1haWwiOiJ0ZXN0QGhlbGl4b25lLmNvbSIsImV4cCI6MTc5MTkzMDA2N30.DDnZTWxmHCfPW6mVJrhKCU0HJeD7vCxcPTTIXwjmq5M")
            return (True, "")

        # === Sinon : Mode normal avec API ===
        try:
            response = requests.post(
                f"{get_api_url()}/auth/login",
                json={"email": email, "password": password},
                timeout=120
            )

            if response.status_code == 200:
                response_data = response.json()
                token = response_data.get("access_token")

                if token:
                    set_auth_token(token)
                    print(f"✅ Authentification réussie pour {email}")
                    return (True, "")
                else:
                    return (False, "Erreur serveur : aucun token reçu.")

            # Erreur d'authentification
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = ""
            if response.status_code == 401:
                return (False, "Email ou mot de passe incorrect.")
            elif response.status_code == 403:
                return (False, detail or "Compte désactivé. Contactez le support.")
            elif response.status_code == 429:
                return (False, "Trop de tentatives. Réessayez dans quelques minutes.")
            else:
                return (False, f"Erreur serveur ({response.status_code}).\n{detail}")

        except requests.exceptions.Timeout:
            return (False, "Le serveur met du temps à répondre.\nRéessayez dans quelques secondes.")
        except requests.exceptions.ConnectionError:
            return (False, "Impossible de contacter le serveur.\nVérifiez votre connexion internet.")
        except Exception as e:
            print(f"[⛔] Erreur serveur : {e}")
            return (False, f"Erreur de connexion au serveur.\n{str(e)}")

    def afficher_indices(self):
        if self._destroyed:
            return
            
        col = 0
        for ticker, name in INDICES.items():
            try:
                label = ctk.CTkLabel(
                    self.top_frame, 
                    text=f"{name}: Chargement...", 
                    text_color="#00BFFF", 
                    font=("Helvetica", 16)
                )
                label.grid(row=0, column=col, padx=15)
                self.indice_labels[ticker] = label
                col += 1
            except Exception as e:
                print(f"[⚠️] Erreur création label {ticker}: {e}")
        
        # Démarrer les mises à jour
        self.start_indices_updates()

    def start_indices_updates(self):
        """Démarre les mises à jour des indices de manière sécurisée"""
        if self._destroyed or self._stop_updates:
            return
        
        def update_worker():
            """Worker thread pour les mises à jour"""
            while not self._stop_updates and not self._destroyed:
                try:
                    self.update_indices_safe()
                    # Attendre 60 secondes avant la prochaine mise à jour
                    for _ in range(600):  # 60 secondes par tranches de 0.1s
                        if self._stop_updates or self._destroyed:
                            return
                        time.sleep(0.1)
                except Exception as e:
                    print(f"[⚠️] Erreur dans update_worker: {e}")
                    break
        
        self._update_thread = threading.Thread(target=update_worker, daemon=True)
        self._update_thread.start()

    def update_indices_safe(self):
        """Mise à jour sécurisée des indices"""
        if self._destroyed or self._stop_updates:
            return

        for ticker, name in INDICES.items():
            if self._destroyed or self._stop_updates:
                break

            try:
                # Vérifier le cache d'abord
                current_time = time.time()
                if ticker in INDICES_CACHE:
                    cached_time, cached_price = INDICES_CACHE[ticker]
                    if current_time - cached_time < CACHE_DURATION:
                        self.update_label_safe(ticker, name, cached_price)
                        continue

                # Récupérer les données avec timeout et retry
                ticker_obj = yf.Ticker(ticker)
                ticker_obj.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })

                data = ticker_obj.history(period="1d", timeout=60)

                if not data.empty:
                    last_price = data["Close"].iloc[-1]
                    # Mettre en cache
                    INDICES_CACHE[ticker] = (current_time, last_price)
                    self.update_label_safe(ticker, name, last_price)
                else:
                    # Utiliser cache si disponible
                    if ticker in INDICES_CACHE:
                        _, cached_price = INDICES_CACHE[ticker]
                        self.update_label_safe(ticker, name, cached_price)
                    else:
                        self.update_label_safe(ticker, name, None)

            except Exception as e:
                print(f"Failed to get ticker '{ticker}' reason: {e}")
                # Utiliser cache si disponible
                if ticker in INDICES_CACHE:
                    _, cached_price = INDICES_CACHE[ticker]
                    self.update_label_safe(ticker, name, cached_price)
                else:
                    self.update_label_safe(ticker, name, None)

    def update_label_safe(self, ticker, name, price):
        """Met à jour un label de manière thread-safe"""
        if self._destroyed or self._stop_updates:
            return
        
        def update_ui():
            try:
                if self._destroyed or self._stop_updates:
                    return
                    
                label = self.indice_labels.get(ticker)
                if label is None:
                    return
                
                # Vérifier que le widget existe encore
                if not self.widget_exists_safe(label):
                    print(f"[⚠️] Widget détruit pour {ticker}")
                    return
                
                # Mettre à jour le texte
                if price is not None:
                    text = f"{name}: {price:.2f}"
                    color = "#00BFFF"
                else:
                    text = f"{name}: --"
                    color = "#666666"

                label.configure(text=text, text_color=color)
                
            except Exception as e:
                print(f"[⚠️] Erreur UI update {ticker}: {e}")
        
        # Programmer la mise à jour sur le thread principal
        try:
            if not self._destroyed and not self._stop_updates:
                self.after(0, update_ui)
        except Exception as e:
            print(f"[⚠️] Erreur programmation UI: {e}")

    def widget_exists_safe(self, widget):
        """Vérifie si un widget existe encore de manière sécurisée"""
        try:
            return widget.winfo_exists()
        except:
            return False

    def destroy(self):
        """Override destroy pour nettoyer proprement"""
        print("[DEBUG] Destruction HomePanel...")
        self._destroyed = True
        self._stop_updates = True
        
        # Attendre que le thread se termine proprement
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        # Nettoyer les références
        self.indice_labels.clear()
        
        try:
            super().destroy()
        except Exception as e:
            print(f"[⚠️] Erreur destruction: {e}")

    def __del__(self):
        """Destructor pour sécurité"""
        self._destroyed = True
        self._stop_updates = True