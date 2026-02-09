"""
HelixOne - Page Profil Utilisateur
Informations personnelles, abonnement, sécurité
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict
from src.i18n import t


class ProfilePanel(ctk.CTkScrollableFrame):
    """Page de profil utilisateur"""

    def __init__(self, parent, user_info: Dict = None, auth_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.user_info = user_info or {}
        self.auth_manager = auth_manager
        self.user_data = self._load_user_data()

        self._create_ui()

    def _load_user_data(self) -> Dict:
        """Charge les données utilisateur locales"""
        try:
            profile_path = Path("data/user_profile.json")
            if profile_path.exists():
                with open(profile_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Erreur chargement profil: {e}")

        return {
            "bio": "",
            "location": "",
        }

    def _save_user_data(self):
        """Sauvegarde les données utilisateur"""
        try:
            profile_path = Path("data/user_profile.json")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")

    def _create_ui(self):
        """Interface principale"""
        # Container central
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.pack(fill="both", expand=True, padx=40, pady=20)

        # Titre de la page
        self._create_page_header(center)

        # Grid de 2 colonnes pour les sections
        main_grid = ctk.CTkFrame(center, fg_color="transparent")
        main_grid.pack(fill="both", expand=True, pady=(20, 0))

        # Colonne gauche
        left_col = ctk.CTkFrame(main_grid, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Colonne droite
        right_col = ctk.CTkFrame(main_grid, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # === SECTIONS COLONNE GAUCHE ===
        self._create_personal_info_section(left_col)
        self._create_subscription_section(left_col)

        # === SECTIONS COLONNE DROITE ===
        self._create_security_section(right_col)
        self._create_danger_zone_section(right_col)

    def _create_page_header(self, parent):
        """Header de la page profil"""
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header,
            text="Mon Profil",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#FFFFFF"
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="Gérez vos informations personnelles",
            font=ctk.CTkFont(size=14),
            text_color="#6b6b6b"
        ).pack(side="left", padx=(15, 0))

    # =========================================================================
    # SECTION 1: INFORMATIONS PERSONNELLES
    # =========================================================================
    def _create_personal_info_section(self, parent):
        """Section informations personnelles"""
        section = self._create_section(parent, "Informations personnelles", "Vos données de compte")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        # Avatar et infos principales
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 20))

        # Avatar
        avatar_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        avatar_frame.pack(side="left")

        avatar = ctk.CTkFrame(
            avatar_frame, width=80, height=80,
            fg_color="#00D9FF", corner_radius=40
        )
        avatar.pack()
        avatar.pack_propagate(False)

        # Vérifier si un avatar personnalisé existe
        avatar_path = self.user_data.get("avatar_path")
        if avatar_path and os.path.exists(avatar_path):
            try:
                avatar_img = Image.open(avatar_path)
                avatar_img = avatar_img.resize((80, 80), Image.Resampling.LANCZOS)
                self.avatar_image = ctk.CTkImage(
                    light_image=avatar_img, dark_image=avatar_img, size=(80, 80)
                )
                ctk.CTkLabel(
                    avatar, image=self.avatar_image, text=""
                ).pack(expand=True)
            except Exception:
                # Fallback aux initiales
                initials = self._get_initials()
                ctk.CTkLabel(
                    avatar, text=initials,
                    font=ctk.CTkFont(size=28, weight="bold"),
                    text_color="#000000"
                ).pack(expand=True)
        else:
            initials = self._get_initials()
            ctk.CTkLabel(
                avatar, text=initials,
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color="#000000"
            ).pack(expand=True)

        ctk.CTkButton(
            avatar_frame, text="Changer",
            font=ctk.CTkFont(size=11),
            fg_color="#2a2d36", hover_color="#3a3d46",
            width=80, height=28, corner_radius=6,
            command=self._change_avatar
        ).pack(pady=(8, 0))

        # Infos à droite de l'avatar
        info_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, padx=(20, 0))

        # Nom
        first = self.user_info.get('first_name', '')
        last = self.user_info.get('last_name', '')
        name = f"{first} {last}".strip() or "Utilisateur"

        ctk.CTkLabel(
            info_frame, text=name,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w")

        # Email
        email = self.user_info.get('email', 'email@example.com')
        ctk.CTkLabel(
            info_frame, text=email,
            font=ctk.CTkFont(size=13),
            text_color="#00D9FF"
        ).pack(anchor="w", pady=(2, 0))

        # Date d'inscription
        joined = self.user_info.get('created_at', datetime.now().strftime("%Y-%m-%d"))
        ctk.CTkLabel(
            info_frame, text=f"Membre depuis {self._format_date(joined)}",
            font=ctk.CTkFont(size=12),
            text_color="#6b6b6b"
        ).pack(anchor="w", pady=(5, 0))

        # Champs éditables
        fields_frame = ctk.CTkFrame(content, fg_color="transparent")
        fields_frame.pack(fill="x")

        # Bio
        ctk.CTkLabel(
            fields_frame, text="Bio",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.bio_entry = ctk.CTkTextbox(
            fields_frame, height=60,
            fg_color="#1a1d24", border_width=1,
            border_color="#2a2d36", corner_radius=8
        )
        self.bio_entry.pack(fill="x", pady=(0, 15))
        self.bio_entry.insert("0.0", self.user_data.get("bio", ""))

        # Localisation
        ctk.CTkLabel(
            fields_frame, text="Localisation",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.location_entry = ctk.CTkEntry(
            fields_frame, height=38,
            fg_color="#1a1d24", border_width=1,
            border_color="#2a2d36", corner_radius=8,
            placeholder_text="Ville, Pays"
        )
        self.location_entry.pack(fill="x", pady=(0, 15))
        self.location_entry.insert(0, self.user_data.get("location", ""))

        # Bouton sauvegarder
        ctk.CTkButton(
            fields_frame, text="Sauvegarder les informations",
            font=ctk.CTkFont(size=13),
            fg_color="#00D9FF", hover_color="#00B8E6",
            text_color="#000000", height=38, corner_radius=8,
            command=self._save_personal_info
        ).pack(anchor="w")

    # =========================================================================
    # SECTION 2: ABONNEMENT & LICENCE
    # =========================================================================
    def _create_subscription_section(self, parent):
        """Section abonnement et licence"""
        section = self._create_section(parent, "Abonnement & Licence", "Votre plan actuel")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        # Récupérer infos licence
        license_type = "FREE"
        days_remaining = 0

        if self.auth_manager:
            try:
                license_info = self.auth_manager.get_license_info()
                license_type = license_info.get('license_type', 'FREE').upper()
                days_remaining = license_info.get('days_remaining', 0)
            except Exception:
                pass

        # Badge du plan
        plan_colors = {
            "FREE": ("#6b6b6b", "#2a2d36"),
            "PREMIUM": ("#FFD700", "#3d3520"),
            "PRO": ("#00D9FF", "#1a3040"),
            "ENTERPRISE": ("#FF6B6B", "#402020")
        }

        text_color, bg_color = plan_colors.get(license_type, ("#6b6b6b", "#2a2d36"))

        plan_frame = ctk.CTkFrame(content, fg_color=bg_color, corner_radius=12)
        plan_frame.pack(fill="x", pady=(0, 15))

        plan_inner = ctk.CTkFrame(plan_frame, fg_color="transparent")
        plan_inner.pack(fill="x", padx=20, pady=20)

        # Icône et nom du plan
        plan_header = ctk.CTkFrame(plan_inner, fg_color="transparent")
        plan_header.pack(fill="x")

        plan_icons = {"FREE": "Plan", "PREMIUM": "Plan", "PRO": "Plan", "ENTERPRISE": "Plan"}

        ctk.CTkLabel(
            plan_header, text=f"{plan_icons.get(license_type, 'Plan')} {license_type}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=text_color
        ).pack(side="left")

        if days_remaining > 0:
            ctk.CTkLabel(
                plan_header, text=f"{days_remaining} jours restants",
                font=ctk.CTkFont(size=13),
                text_color="#AAAAAA"
            ).pack(side="right")

        # Fonctionnalités du plan
        features_frame = ctk.CTkFrame(plan_inner, fg_color="transparent")
        features_frame.pack(fill="x", pady=(15, 0))

        features = {
            "FREE": ["5 analyses/jour", "Alertes basiques", "1 portefeuille"],
            "PREMIUM": ["Analyses illimitées", "Alertes avancées", "5 portefeuilles", "Support prioritaire"],
            "PRO": ["Tout Premium +", "API Access", "Portefeuilles illimités", "ML Predictions", "Support 24/7"],
        }

        for feature in features.get(license_type, features["FREE"]):
            feat_row = ctk.CTkFrame(features_frame, fg_color="transparent")
            feat_row.pack(fill="x", pady=2)

            ctk.CTkLabel(
                feat_row, text="v",
                font=ctk.CTkFont(size=12),
                text_color="#00FF88"
            ).pack(side="left")

            ctk.CTkLabel(
                feat_row, text=feature,
                font=ctk.CTkFont(size=12),
                text_color="#CCCCCC"
            ).pack(side="left", padx=(8, 0))

        # Bouton upgrade
        if license_type == "FREE":
            ctk.CTkButton(
                content, text="Passer à Premium",
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color="#FFD700", hover_color="#E6C200",
                text_color="#000000", height=42, corner_radius=8,
                command=self._upgrade_plan
            ).pack(fill="x")

    # =========================================================================
    # SECTION 3: SECURITE
    # =========================================================================
    def _create_security_section(self, parent):
        """Section sécurité"""
        section = self._create_section(parent, "Sécurité", "Protégez votre compte")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        # Changer mot de passe
        pwd_frame = ctk.CTkFrame(content, fg_color="#1a1d24", corner_radius=8)
        pwd_frame.pack(fill="x", pady=(0, 10))

        pwd_inner = ctk.CTkFrame(pwd_frame, fg_color="transparent")
        pwd_inner.pack(fill="x", padx=15, pady=15)

        pwd_info = ctk.CTkFrame(pwd_inner, fg_color="transparent")
        pwd_info.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            pwd_info, text="Mot de passe",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            pwd_info, text="Cliquez pour modifier votre mot de passe",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b"
        ).pack(anchor="w")

        ctk.CTkButton(
            pwd_inner, text="Modifier",
            font=ctk.CTkFont(size=12),
            fg_color="#2a2d36", hover_color="#3a3d46",
            width=90, height=32, corner_radius=6,
            command=self._change_password
        ).pack(side="right")

    # =========================================================================
    # SECTION 4: ZONE DANGER
    # =========================================================================
    def _create_danger_zone_section(self, parent):
        """Section zone de danger"""
        section = ctk.CTkFrame(parent, fg_color="#1a1d24", corner_radius=12)
        section.pack(fill="x", pady=(20, 0))

        # Header rouge
        header = ctk.CTkFrame(section, fg_color="#401515", corner_radius=12)
        header.pack(fill="x")

        header_inner = ctk.CTkFrame(header, fg_color="transparent")
        header_inner.pack(fill="x", padx=20, pady=15)

        ctk.CTkLabel(
            header_inner, text="Zone de danger",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FF6B6B"
        ).pack(side="left")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=20)

        # Déconnexion
        logout_frame = ctk.CTkFrame(content, fg_color="transparent")
        logout_frame.pack(fill="x", pady=(0, 15))

        logout_info = ctk.CTkFrame(logout_frame, fg_color="transparent")
        logout_info.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            logout_info, text="Se déconnecter",
            font=ctk.CTkFont(size=13),
            text_color="#FFFFFF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            logout_info, text="Vous serez redirigé vers la page de connexion",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b"
        ).pack(anchor="w")

        ctk.CTkButton(
            logout_frame, text="Déconnexion",
            font=ctk.CTkFont(size=12),
            fg_color="#2a2d36", hover_color="#3a3d46",
            width=110, height=32, corner_radius=6,
            command=self._logout
        ).pack(side="right")

        # Supprimer compte
        delete_frame = ctk.CTkFrame(content, fg_color="transparent")
        delete_frame.pack(fill="x")

        delete_info = ctk.CTkFrame(delete_frame, fg_color="transparent")
        delete_info.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            delete_info, text="Supprimer le compte",
            font=ctk.CTkFont(size=13),
            text_color="#FF6B6B"
        ).pack(anchor="w")

        ctk.CTkLabel(
            delete_info, text="Cette action est irréversible",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b"
        ).pack(anchor="w")

        ctk.CTkButton(
            delete_frame, text="Supprimer",
            font=ctk.CTkFont(size=12),
            fg_color="#FF4444", hover_color="#CC3333",
            text_color="#FFFFFF",
            width=110, height=32, corner_radius=6,
            command=self._delete_account
        ).pack(side="right")

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _create_section(self, parent, title: str, subtitle: str) -> ctk.CTkFrame:
        """Crée une section avec header"""
        section = ctk.CTkFrame(parent, fg_color="#1a1d24", corner_radius=12)
        section.pack(fill="x", pady=(0, 15))

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 15))

        ctk.CTkLabel(
            header, text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            header, text=subtitle,
            font=ctk.CTkFont(size=12),
            text_color="#6b6b6b"
        ).pack(anchor="w")

        return section

    def _get_initials(self) -> str:
        first = self.user_info.get('first_name', '')
        last = self.user_info.get('last_name', '')
        email = self.user_info.get('email', '')

        if first:
            return f"{first[0]}{last[0] if last else ''}".upper()
        elif email:
            return email[0].upper()
        return "U"

    def _format_date(self, date_str: str) -> str:
        try:
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            months = ["janvier", "février", "mars", "avril", "mai", "juin",
                      "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
            return f"{months[dt.month - 1]} {dt.year}"
        except Exception:
            return date_str

    # =========================================================================
    # ACTIONS
    # =========================================================================
    def _change_avatar(self):
        """Change l'avatar avec sélection d'image"""
        file_path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Tous les fichiers", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            img = Image.open(file_path)

            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Redimensionner en carré (crop center)
            min_dim = min(img.size)
            left = (img.width - min_dim) // 2
            top = (img.height - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            img = img.resize((200, 200), Image.Resampling.LANCZOS)

            # Sauvegarder localement
            avatar_dir = Path("data/avatars")
            avatar_dir.mkdir(parents=True, exist_ok=True)

            avatar_path = avatar_dir / "user_avatar.png"
            img.save(avatar_path, "PNG")

            self.user_data["avatar_path"] = str(avatar_path)
            self._save_user_data()

            messagebox.showinfo("Avatar", "Avatar mis à jour !\nRechargez la page pour voir les changements.")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image:\n{str(e)}")

    def _save_personal_info(self):
        """Sauvegarde les infos personnelles"""
        self.user_data["bio"] = self.bio_entry.get("0.0", "end").strip()
        self.user_data["location"] = self.location_entry.get().strip()
        self._save_user_data()
        messagebox.showinfo("Succès", "Informations sauvegardées !")

    def _upgrade_plan(self):
        """Upgrade du plan"""
        messagebox.showinfo("Upgrade", "Contactez-nous pour passer à Premium :\nsupport@helixone.com")

    def _change_password(self):
        """Ouvre le dialogue de changement de mot de passe"""
        ChangePasswordDialog(self, self.auth_manager)

    def _logout(self):
        """Déconnexion"""
        if messagebox.askyesno("Déconnexion", "Voulez-vous vous déconnecter ?"):
            if self.auth_manager:
                self.auth_manager.logout()
            import sys
            sys.exit(0)

    def _delete_account(self):
        """Suppression du compte"""
        print("[DEBUG] _delete_account appelé")

        if not self.auth_manager:
            print("[DEBUG] auth_manager est None!")
            messagebox.showerror("Erreur", "Connexion requise")
            return

        if not hasattr(self.auth_manager, 'client'):
            print("[DEBUG] auth_manager n'a pas d'attribut 'client'!")
            messagebox.showerror("Erreur", "Client API non initialisé")
            return

        try:
            print("[DEBUG] Ouverture du dialogue de suppression...")
            dialog = DeleteAccountDialog(self, self.auth_manager)
            dialog.focus_force()  # Force le focus sur le dialogue
            self.wait_window(dialog)
            print("[DEBUG] Dialogue fermé")
        except Exception as e:
            print(f"[DEBUG] Erreur ouverture dialogue: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le dialogue:\n{str(e)}")


class ChangePasswordDialog(ctk.CTkToplevel):
    """Dialogue de changement de mot de passe"""

    def __init__(self, parent, auth_manager=None):
        super().__init__(parent)

        self.auth_manager = auth_manager
        self.title("Changer le mot de passe")
        self.geometry("400x350")
        self.configure(fg_color="#0f1117")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._create_ui()

        # Centrer
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 350) // 2
        self.geometry(f"+{x}+{y}")

    def _create_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1d24", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="Changer le mot de passe",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFFFFF"
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkButton(
            header, text="X", width=30, height=30,
            fg_color="transparent", hover_color="#2a2d36",
            command=self.destroy
        ).pack(side="right", padx=10, pady=10)

        # Form
        form = ctk.CTkFrame(self, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=25, pady=20)

        # Ancien mot de passe
        ctk.CTkLabel(
            form, text="Mot de passe actuel",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.old_pwd = ctk.CTkEntry(
            form, show="*", height=40,
            fg_color="#1a1d24", border_color="#2a2d36"
        )
        self.old_pwd.pack(fill="x", pady=(0, 15))

        # Nouveau mot de passe
        ctk.CTkLabel(
            form, text="Nouveau mot de passe",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.new_pwd = ctk.CTkEntry(
            form, show="*", height=40,
            fg_color="#1a1d24", border_color="#2a2d36"
        )
        self.new_pwd.pack(fill="x", pady=(0, 15))

        # Confirmer
        ctk.CTkLabel(
            form, text="Confirmer le mot de passe",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.confirm_pwd = ctk.CTkEntry(
            form, show="*", height=40,
            fg_color="#1a1d24", border_color="#2a2d36"
        )
        self.confirm_pwd.pack(fill="x", pady=(0, 20))

        # Boutons
        btn_frame = ctk.CTkFrame(form, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame, text="Annuler",
            fg_color="#2a2d36", hover_color="#3a3d46",
            height=40, corner_radius=8,
            command=self.destroy
        ).pack(side="left", expand=True, padx=(0, 5))

        ctk.CTkButton(
            btn_frame, text="Changer",
            fg_color="#00D9FF", hover_color="#00B8E6",
            text_color="#000000",
            height=40, corner_radius=8,
            command=self._change
        ).pack(side="right", expand=True, padx=(5, 0))

    def _change(self):
        old = self.old_pwd.get()
        new = self.new_pwd.get()
        confirm = self.confirm_pwd.get()

        if not all([old, new, confirm]):
            messagebox.showerror("Erreur", "Veuillez remplir tous les champs")
            return

        if new != confirm:
            messagebox.showerror("Erreur", "Les mots de passe ne correspondent pas")
            return

        if len(new) < 8:
            messagebox.showerror("Erreur", "Le mot de passe doit faire au moins 8 caractères")
            return

        # Appeler l'API pour changer le mot de passe
        if self.auth_manager and hasattr(self.auth_manager, 'client'):
            try:
                result = self.auth_manager.client.change_password(old, new)
                if result.get("success"):
                    messagebox.showinfo("Succès", "Mot de passe modifié avec succès !")
                    self.destroy()
                else:
                    messagebox.showerror("Erreur", result.get("message", "Erreur inconnue"))
            except Exception as e:
                error_msg = str(e)
                if "incorrect" in error_msg.lower():
                    messagebox.showerror("Erreur", "Mot de passe actuel incorrect")
                else:
                    messagebox.showerror("Erreur", f"Erreur: {error_msg}")
        else:
            messagebox.showerror("Erreur", "Connexion requise")
            self.destroy()


class DeleteAccountDialog(ctk.CTkToplevel):
    """Dialogue de suppression de compte"""

    def __init__(self, parent, auth_manager):
        super().__init__(parent)

        self.auth_manager = auth_manager

        self.title("Supprimer le compte")
        self.geometry("450x400")
        self.configure(fg_color="#0f1117")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._create_ui()

        # Centrer
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 450) // 2
        y = (self.winfo_screenheight() - 400) // 2
        self.geometry(f"+{x}+{y}")

    def _create_ui(self):
        # Header rouge danger
        header = ctk.CTkFrame(self, fg_color="#401515", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="SUPPRIMER LE COMPTE",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FF4444"
        ).pack(side="left", padx=20, pady=15)

        ctk.CTkButton(
            header, text="X", width=35, height=35,
            fg_color="transparent", hover_color="#2a2d36",
            command=self.destroy
        ).pack(side="right", padx=15, pady=12)

        # Content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=25, pady=20)

        # Avertissement
        warning_frame = ctk.CTkFrame(content, fg_color="#2a1515", corner_radius=10)
        warning_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            warning_frame,
            text="ATTENTION - ACTION IRREVERSIBLE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FF6B6B"
        ).pack(padx=15, pady=(15, 5))

        ctk.CTkLabel(
            warning_frame,
            text="Cette action supprimera définitivement:\n- Votre compte et profil\n- Toutes vos données et préférences\n- Votre licence et abonnement",
            font=ctk.CTkFont(size=12),
            text_color="#CCAAAA",
            justify="left"
        ).pack(padx=15, pady=(5, 15))

        # Mot de passe
        ctk.CTkLabel(
            content, text="Entrez votre mot de passe:",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.password_entry = ctk.CTkEntry(
            content, show="*", height=40,
            fg_color="#1a1d24", border_color="#2a2d36"
        )
        self.password_entry.pack(fill="x", pady=(0, 15))

        # Confirmation
        ctk.CTkLabel(
            content, text="Tapez SUPPRIMER pour confirmer:",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 5))

        self.confirm_entry = ctk.CTkEntry(
            content, height=40,
            fg_color="#1a1d24", border_color="#2a2d36",
            placeholder_text="SUPPRIMER"
        )
        self.confirm_entry.pack(fill="x", pady=(0, 10))

        # Error label
        self.error_label = ctk.CTkLabel(
            content,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#FF4444"
        )
        self.error_label.pack(pady=(0, 10))

        # Boutons
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame, text="Annuler",
            fg_color="#2a2d36", hover_color="#3a3d46",
            height=42, corner_radius=8,
            command=self.destroy
        ).pack(side="left", expand=True, padx=(0, 5))

        self.delete_btn = ctk.CTkButton(
            btn_frame, text="Supprimer définitivement",
            fg_color="#FF4444", hover_color="#CC3333",
            text_color="#FFFFFF",
            height=42, corner_radius=8,
            command=self._delete
        )
        self.delete_btn.pack(side="right", expand=True, padx=(5, 0))

    def _delete(self):
        password = self.password_entry.get()
        confirm = self.confirm_entry.get()

        if not password:
            self.error_label.configure(text="Entrez votre mot de passe")
            return

        if confirm != "SUPPRIMER":
            self.error_label.configure(text="Tapez exactement 'SUPPRIMER'")
            return

        try:
            self.delete_btn.configure(state="disabled", text="Suppression...")
            self.update()

            result = self.auth_manager.client.delete_account(password, confirm)

            if result.get("success"):
                messagebox.showinfo(
                    "Compte supprimé",
                    "Votre compte a été supprimé définitivement.\n"
                    "L'application va se fermer."
                )
                self.destroy()
                import sys
                sys.exit(0)
            else:
                self.error_label.configure(text=result.get("message", "Erreur"))
                self.delete_btn.configure(state="normal", text="Supprimer définitivement")

        except Exception as e:
            error_msg = str(e)
            if "incorrect" in error_msg.lower():
                self.error_label.configure(text="Mot de passe incorrect")
            else:
                self.error_label.configure(text=f"Erreur: {error_msg}")
            self.delete_btn.configure(state="normal", text="Supprimer définitivement")
