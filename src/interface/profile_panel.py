"""
HelixOne - Page Profil Utilisateur Complète
Toutes les fonctionnalités de personnalisation
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import json
import os
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from src.i18n import t


class ProfilePanel(ctk.CTkScrollableFrame):
    """Page de profil utilisateur complète"""

    def __init__(self, parent, user_info: Dict = None, auth_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.user_info = user_info or {}
        self.auth_manager = auth_manager
        self.user_data = self._load_user_data()
        self.preferences = self._load_preferences()

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
            "risk_tolerance": "moderate",
            "favorite_markets": ["stocks"],
            "default_currency": "EUR",
            "email_alerts": True,
            "price_alerts": True,
            "weekly_report": False,
            "total_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }

    def _load_preferences(self) -> Dict:
        """Charge les préférences de trading"""
        try:
            pref_path = Path("data/trading_preferences.json")
            if pref_path.exists():
                with open(pref_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "risk_tolerance": "moderate",
            "favorite_markets": ["stocks"],
            "default_currency": "EUR"
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

    def _save_preferences(self):
        """Sauvegarde les préférences"""
        try:
            pref_path = Path("data/trading_preferences.json")
            pref_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pref_path, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde préférences: {e}")

    def _create_ui(self):
        """Interface principale avec toutes les sections"""
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
        self._create_statistics_section(left_col)
        self._create_connections_section(left_col)

        # === SECTIONS COLONNE DROITE ===
        self._create_trading_preferences_section(right_col)
        self._create_notifications_section(right_col)
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
            text="Gérez vos informations et préférences",
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

        plan_icons = {"FREE": "🆓", "PREMIUM": "⭐", "PRO": "💎", "ENTERPRISE": "🏢"}

        ctk.CTkLabel(
            plan_header, text=f"{plan_icons.get(license_type, '📦')} Plan {license_type}",
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
                feat_row, text="✓",
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
    # SECTION 3: PREFERENCES DE TRADING
    # =========================================================================
    def _create_trading_preferences_section(self, parent):
        """Section préférences de trading"""
        section = self._create_section(parent, "Préférences de Trading", "Personnalisez votre expérience")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        # Tolérance au risque
        ctk.CTkLabel(
            content, text="Tolérance au risque",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 8))

        risk_frame = ctk.CTkFrame(content, fg_color="transparent")
        risk_frame.pack(fill="x", pady=(0, 20))

        self.risk_var = ctk.StringVar(value=self.preferences.get("risk_tolerance", "moderate"))

        risks = [
            ("conservative", "Conservateur", "Faible risque, rendements stables"),
            ("moderate", "Modéré", "Équilibre risque/rendement"),
            ("aggressive", "Agressif", "Haut risque, hauts rendements potentiels")
        ]

        for value, label, desc in risks:
            risk_option = ctk.CTkFrame(risk_frame, fg_color="#1a1d24", corner_radius=8)
            risk_option.pack(fill="x", pady=3)

            radio = ctk.CTkRadioButton(
                risk_option, text="",
                variable=self.risk_var, value=value,
                fg_color="#00D9FF", hover_color="#00B8E6",
                command=self._on_risk_change
            )
            radio.pack(side="left", padx=(15, 10), pady=12)

            info = ctk.CTkFrame(risk_option, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True)

            ctk.CTkLabel(
                info, text=label,
                font=ctk.CTkFont(size=13),
                text_color="#FFFFFF"
            ).pack(anchor="w")

            ctk.CTkLabel(
                info, text=desc,
                font=ctk.CTkFont(size=11),
                text_color="#6b6b6b"
            ).pack(anchor="w")

        # Marchés favoris
        ctk.CTkLabel(
            content, text="Marchés favoris",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(10, 8))

        markets_frame = ctk.CTkFrame(content, fg_color="transparent")
        markets_frame.pack(fill="x", pady=(0, 20))

        self.market_vars = {}
        markets = [
            ("stocks", "Actions", "📈"),
            ("crypto", "Crypto", "🪙"),
            ("forex", "Forex", "💱"),
            ("commodities", "Matières premières", "🛢️"),
            ("etf", "ETF", "📊")
        ]

        current_markets = self.preferences.get("favorite_markets", ["stocks"])

        for i, (value, label, icon) in enumerate(markets):
            self.market_vars[value] = ctk.BooleanVar(value=value in current_markets)

            market_chip = ctk.CTkCheckBox(
                markets_frame, text=f"{icon} {label}",
                variable=self.market_vars[value],
                font=ctk.CTkFont(size=12),
                fg_color="#00D9FF", hover_color="#00B8E6",
                corner_radius=6,
                command=self._on_markets_change
            )
            market_chip.pack(side="left", padx=(0 if i == 0 else 10, 0))

        # Devise par défaut
        ctk.CTkLabel(
            content, text="Devise par défaut",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(10, 8))

        self.currency_var = ctk.StringVar(value=self.preferences.get("default_currency", "EUR"))

        currency_menu = ctk.CTkOptionMenu(
            content,
            variable=self.currency_var,
            values=["EUR", "USD", "GBP", "CHF", "JPY"],
            font=ctk.CTkFont(size=13),
            fg_color="#1a1d24",
            button_color="#2a2d36",
            button_hover_color="#3a3d46",
            dropdown_fg_color="#1a1d24",
            dropdown_hover_color="#2a2d36",
            width=150, height=38,
            command=self._on_currency_change
        )
        currency_menu.pack(anchor="w")

    # =========================================================================
    # SECTION 4: STATISTIQUES
    # =========================================================================
    def _create_statistics_section(self, parent):
        """Section statistiques et badges"""
        section = self._create_section(parent, "Statistiques & Achievements", "Votre progression")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        # Stats grid
        stats_grid = ctk.CTkFrame(content, fg_color="transparent")
        stats_grid.pack(fill="x", pady=(0, 20))

        stats = [
            ("Analyses", str(self.user_data.get("total_analyses", 0)), "#00D9FF"),
            ("Win Rate", f"{self.user_data.get('win_rate', 0):.0f}%", "#00FF88"),
            ("Rendement", f"{self.user_data.get('total_return', 0):+.1f}%",
             "#00FF88" if self.user_data.get('total_return', 0) >= 0 else "#FF4444"),
            ("Trades", str(self.user_data.get("total_trades", 0)), "#FFD700"),
        ]

        for i, (label, value, color) in enumerate(stats):
            stat_card = ctk.CTkFrame(stats_grid, fg_color="#1a1d24", corner_radius=10)
            stat_card.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 5, 0))

            ctk.CTkLabel(
                stat_card, text=value,
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color=color
            ).pack(pady=(15, 2))

            ctk.CTkLabel(
                stat_card, text=label,
                font=ctk.CTkFont(size=11),
                text_color="#6b6b6b"
            ).pack(pady=(0, 15))

        # Badges
        ctk.CTkLabel(
            content, text="Badges obtenus",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(0, 10))

        badges_frame = ctk.CTkFrame(content, fg_color="transparent")
        badges_frame.pack(fill="x")

        badges = [
            ("🌱", "Premier pas", True),
            ("📚", "Studieux", True),
            ("🎯", "Précision", False),
            ("💎", "Diamant", False),
            ("🏆", "Champion", False),
            ("🔥", "En feu", False),
        ]

        for icon, name, unlocked in badges:
            badge = ctk.CTkFrame(
                badges_frame,
                fg_color="#1a1d24" if unlocked else "#0f1117",
                corner_radius=8
            )
            badge.pack(side="left", padx=(0, 8), pady=4)

            opacity = 1.0 if unlocked else 0.3

            ctk.CTkLabel(
                badge, text=f"{icon} {name}",
                font=ctk.CTkFont(size=11),
                text_color="#FFFFFF" if unlocked else "#4a4a4a"
            ).pack(padx=12, pady=8)

    # =========================================================================
    # SECTION 5: NOTIFICATIONS
    # =========================================================================
    def _create_notifications_section(self, parent):
        """Section notifications"""
        section = self._create_section(parent, "Notifications", "Gérez vos alertes")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        notifications = [
            ("email_alerts", "Alertes par email", "Recevez les alertes importantes par email", True),
            ("price_alerts", "Alertes de prix", "Notifications quand vos prix cibles sont atteints", True),
            ("weekly_report", "Rapport hebdomadaire", "Résumé de votre activité chaque semaine", False),
            ("news_alerts", "Actualités", "Alertes sur les news importantes du marché", False),
            ("ml_signals", "Signaux ML", "Notifications des prédictions ML", True),
        ]

        self.notif_vars = {}

        for key, label, desc, default in notifications:
            current_val = self.user_data.get(key, default)
            self.notif_vars[key] = ctk.BooleanVar(value=current_val)

            notif_row = ctk.CTkFrame(content, fg_color="#1a1d24", corner_radius=8)
            notif_row.pack(fill="x", pady=3)

            info = ctk.CTkFrame(notif_row, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True, padx=15, pady=12)

            ctk.CTkLabel(
                info, text=label,
                font=ctk.CTkFont(size=13),
                text_color="#FFFFFF"
            ).pack(anchor="w")

            ctk.CTkLabel(
                info, text=desc,
                font=ctk.CTkFont(size=11),
                text_color="#6b6b6b"
            ).pack(anchor="w")

            switch = ctk.CTkSwitch(
                notif_row, text="",
                variable=self.notif_vars[key],
                fg_color="#2a2d36",
                progress_color="#00D9FF",
                button_color="#FFFFFF",
                button_hover_color="#EEEEEE",
                command=lambda k=key: self._on_notif_change(k)
            )
            switch.pack(side="right", padx=15)

    # =========================================================================
    # SECTION 6: SECURITE
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
            pwd_info, text="🔒 Mot de passe",
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

        # 2FA - Charger le statut réel depuis l'API
        twofa_enabled = False
        if self.auth_manager and hasattr(self.auth_manager, 'client'):
            try:
                status = self.auth_manager.client.get_2fa_status()
                twofa_enabled = status.get("enabled", False)
            except Exception:
                pass

        twofa_frame = ctk.CTkFrame(content, fg_color="#1a1d24", corner_radius=8)
        twofa_frame.pack(fill="x", pady=(0, 10))

        twofa_inner = ctk.CTkFrame(twofa_frame, fg_color="transparent")
        twofa_inner.pack(fill="x", padx=15, pady=15)

        twofa_info = ctk.CTkFrame(twofa_inner, fg_color="transparent")
        twofa_info.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            twofa_info, text="🛡️ Authentification 2FA",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w")

        if twofa_enabled:
            status_text = "✅ Activée - Votre compte est protégé"
            status_color = "#00FF88"
            btn_text = "Désactiver"
            btn_fg = "#FF6B6B"
            btn_hover = "#CC5555"
            btn_text_color = "#FFFFFF"
            btn_command = self._disable_2fa
        else:
            status_text = "⚠️ Non activée - Recommandé pour plus de sécurité"
            status_color = "#FFD700"
            btn_text = "Activer"
            btn_fg = "#00D9FF"
            btn_hover = "#00B8E6"
            btn_text_color = "#000000"
            btn_command = self._enable_2fa

        ctk.CTkLabel(
            twofa_info, text=status_text,
            font=ctk.CTkFont(size=11),
            text_color=status_color
        ).pack(anchor="w")

        ctk.CTkButton(
            twofa_inner, text=btn_text,
            font=ctk.CTkFont(size=12),
            fg_color=btn_fg, hover_color=btn_hover,
            text_color=btn_text_color,
            width=90, height=32, corner_radius=6,
            command=btn_command
        ).pack(side="right")

        # Sessions actives
        sessions_frame = ctk.CTkFrame(content, fg_color="#1a1d24", corner_radius=8)
        sessions_frame.pack(fill="x")

        sessions_inner = ctk.CTkFrame(sessions_frame, fg_color="transparent")
        sessions_inner.pack(fill="x", padx=15, pady=15)

        sessions_info = ctk.CTkFrame(sessions_inner, fg_color="transparent")
        sessions_info.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            sessions_info, text="📱 Session active",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            sessions_info, text="Connecté sur cet appareil",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b"
        ).pack(anchor="w")

    # =========================================================================
    # SECTION 7: CONNEXIONS
    # =========================================================================
    def _create_connections_section(self, parent):
        """Section connexions (IBKR, etc.)"""
        section = self._create_section(parent, "Connexions", "Vos comptes liés")

        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=20, pady=(0, 20))

        connections = [
            ("IBKR", "Interactive Brokers", "Non connecté", False, "#FF6B6B"),
            ("TradingView", "TradingView", "Non connecté", False, "#6b6b6b"),
        ]

        for name, full_name, status, connected, color in connections:
            conn_frame = ctk.CTkFrame(content, fg_color="#1a1d24", corner_radius=8)
            conn_frame.pack(fill="x", pady=3)

            conn_inner = ctk.CTkFrame(conn_frame, fg_color="transparent")
            conn_inner.pack(fill="x", padx=15, pady=15)

            # Logo/Icon
            icon_frame = ctk.CTkFrame(
                conn_inner, width=40, height=40,
                fg_color="#2a2d36", corner_radius=8
            )
            icon_frame.pack(side="left")
            icon_frame.pack_propagate(False)

            icons = {"IBKR": "🏦", "TradingView": "📊"}
            ctk.CTkLabel(
                icon_frame, text=icons.get(name, "🔗"),
                font=ctk.CTkFont(size=18)
            ).pack(expand=True)

            # Info
            info = ctk.CTkFrame(conn_inner, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True, padx=(12, 0))

            ctk.CTkLabel(
                info, text=full_name,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#FFFFFF"
            ).pack(anchor="w")

            status_color = "#00FF88" if connected else color
            ctk.CTkLabel(
                info, text=f"● {status}",
                font=ctk.CTkFont(size=11),
                text_color=status_color
            ).pack(anchor="w")

            # Bouton
            btn_text = "Déconnecter" if connected else "Connecter"
            btn_color = "#2a2d36" if connected else "#00D9FF"
            text_color = "#FFFFFF" if connected else "#000000"

            ctk.CTkButton(
                conn_inner, text=btn_text,
                font=ctk.CTkFont(size=12),
                fg_color=btn_color,
                hover_color="#3a3d46" if connected else "#00B8E6",
                text_color=text_color,
                width=100, height=32, corner_radius=6,
                command=lambda n=name: self._toggle_connection(n)
            ).pack(side="right")

    # =========================================================================
    # SECTION 8: ZONE DANGER
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
            header_inner, text="⚠️ Zone de danger",
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
        # Ouvrir le sélecteur de fichier
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
            # Charger et redimensionner l'image
            img = Image.open(file_path)

            # Convertir en RGB si nécessaire
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

            # Sauvegarder le chemin dans user_data
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

    def _on_risk_change(self):
        """Changement de tolérance au risque"""
        self.preferences["risk_tolerance"] = self.risk_var.get()
        self._save_preferences()

    def _on_markets_change(self):
        """Changement des marchés favoris"""
        self.preferences["favorite_markets"] = [
            k for k, v in self.market_vars.items() if v.get()
        ]
        self._save_preferences()

    def _on_currency_change(self, value):
        """Changement de devise"""
        self.preferences["default_currency"] = value
        self._save_preferences()

    def _on_notif_change(self, key):
        """Changement d'une notification"""
        self.user_data[key] = self.notif_vars[key].get()
        self._save_user_data()

    def _change_password(self):
        """Ouvre le dialogue de changement de mot de passe"""
        ChangePasswordDialog(self, self.auth_manager)

    def _enable_2fa(self):
        """Active le 2FA"""
        if self.auth_manager:
            TwoFASetupDialog(self, self.auth_manager)
        else:
            messagebox.showerror("Erreur", "Connexion requise pour configurer le 2FA")

    def _disable_2fa(self):
        """Désactive le 2FA"""
        if not self.auth_manager:
            messagebox.showerror("Erreur", "Connexion requise")
            return

        # Demander le code TOTP actuel
        dialog = Disable2FADialog(self, self.auth_manager)
        self.wait_window(dialog)

    def _toggle_connection(self, name: str):
        """Connecte/déconnecte un service"""
        if name == "IBKR":
            messagebox.showinfo("IBKR", "Allez dans l'onglet IBKR pour configurer la connexion")
        else:
            messagebox.showinfo(name, f"Connexion {name} à venir")

    def _logout(self):
        """Déconnexion"""
        if messagebox.askyesno("Déconnexion", "Voulez-vous vous déconnecter ?"):
            if self.auth_manager:
                self.auth_manager.logout()
            # Redémarrer l'app
            import sys
            sys.exit(0)

    def _delete_account(self):
        """Suppression du compte"""
        if not self.auth_manager:
            messagebox.showerror("Erreur", "Connexion requise")
            return

        # Ouvrir le dialogue de suppression
        dialog = DeleteAccountDialog(self, self.auth_manager)
        self.wait_window(dialog)


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
            header, text="🔒 Changer le mot de passe",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFFFFF"
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkButton(
            header, text="✕", width=30, height=30,
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
            form, show="•", height=40,
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
            form, show="•", height=40,
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
            form, show="•", height=40,
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


class TwoFASetupDialog(ctk.CTkToplevel):
    """Dialogue de configuration 2FA avec QR Code"""

    def __init__(self, parent, auth_manager):
        super().__init__(parent)

        self.auth_manager = auth_manager
        self.qr_image = None
        self.secret = None

        self.title("Configuration 2FA")
        self.geometry("500x650")
        self.configure(fg_color="#0f1117")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._create_ui()
        self._load_2fa_setup()

        # Centrer
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 500) // 2
        y = (self.winfo_screenheight() - 650) // 2
        self.geometry(f"+{x}+{y}")

    def _create_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1d24", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🛡️ Configuration 2FA",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FFFFFF"
        ).pack(side="left", padx=25, pady=15)

        ctk.CTkButton(
            header, text="✕", width=35, height=35,
            fg_color="transparent", hover_color="#2a2d36",
            command=self.destroy
        ).pack(side="right", padx=15, pady=12)

        # Content
        self.content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=25, pady=20)

        # Instructions
        ctk.CTkLabel(
            self.content,
            text="Protégez votre compte avec l'authentification à deux facteurs",
            font=ctk.CTkFont(size=14),
            text_color="#AAAAAA",
            wraplength=400
        ).pack(anchor="w", pady=(0, 20))

        # Étape 1
        step1 = ctk.CTkFrame(self.content, fg_color="#1a1d24", corner_radius=10)
        step1.pack(fill="x", pady=(0, 15))

        step1_inner = ctk.CTkFrame(step1, fg_color="transparent")
        step1_inner.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            step1_inner,
            text="1️⃣ Téléchargez une app d'authentification",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00D9FF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            step1_inner,
            text="• Google Authenticator\n• Microsoft Authenticator\n• Authy",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA",
            justify="left"
        ).pack(anchor="w", pady=(10, 0))

        # Étape 2 - QR Code
        step2 = ctk.CTkFrame(self.content, fg_color="#1a1d24", corner_radius=10)
        step2.pack(fill="x", pady=(0, 15))

        step2_inner = ctk.CTkFrame(step2, fg_color="transparent")
        step2_inner.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            step2_inner,
            text="2️⃣ Scannez ce QR code avec l'app",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00D9FF"
        ).pack(anchor="w")

        # QR Code placeholder
        self.qr_frame = ctk.CTkFrame(step2_inner, width=200, height=200, fg_color="#FFFFFF", corner_radius=10)
        self.qr_frame.pack(pady=15)
        self.qr_frame.pack_propagate(False)

        self.qr_label = ctk.CTkLabel(
            self.qr_frame,
            text="Chargement...",
            font=ctk.CTkFont(size=12),
            text_color="#000000"
        )
        self.qr_label.pack(expand=True)

        # Secret manuel
        self.secret_frame = ctk.CTkFrame(step2_inner, fg_color="#2a2d36", corner_radius=8)
        self.secret_frame.pack(fill="x", pady=(10, 0))

        self.secret_label = ctk.CTkLabel(
            self.secret_frame,
            text="Ou entrez ce code manuellement:",
            font=ctk.CTkFont(size=11),
            text_color="#AAAAAA"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.secret_code = ctk.CTkLabel(
            self.secret_frame,
            text="...",
            font=ctk.CTkFont(size=14, weight="bold", family="Courier"),
            text_color="#00D9FF"
        )
        self.secret_code.pack(anchor="w", padx=15, pady=(0, 10))

        # Étape 3 - Vérification
        step3 = ctk.CTkFrame(self.content, fg_color="#1a1d24", corner_radius=10)
        step3.pack(fill="x", pady=(0, 15))

        step3_inner = ctk.CTkFrame(step3, fg_color="transparent")
        step3_inner.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            step3_inner,
            text="3️⃣ Entrez le code à 6 chiffres",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00D9FF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            step3_inner,
            text="Entrez le code affiché dans l'app pour confirmer:",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        ).pack(anchor="w", pady=(10, 10))

        self.code_entry = ctk.CTkEntry(
            step3_inner,
            height=50,
            font=ctk.CTkFont(size=24, weight="bold", family="Courier"),
            fg_color="#2a2d36",
            border_color="#3a3d46",
            justify="center",
            placeholder_text="000000"
        )
        self.code_entry.pack(fill="x")
        self.code_entry.bind("<Return>", lambda e: self._verify())

        # Error message
        self.error_label = ctk.CTkLabel(
            self.content,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#FF4444"
        )
        self.error_label.pack(pady=(0, 10))

        # Bouton de confirmation
        self.verify_btn = ctk.CTkButton(
            self.content,
            text="Activer le 2FA",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#00D9FF",
            hover_color="#00B8E6",
            text_color="#000000",
            height=45,
            corner_radius=8,
            command=self._verify
        )
        self.verify_btn.pack(fill="x")

    def _load_2fa_setup(self):
        """Charge le QR code depuis l'API"""
        try:
            # Appeler l'API pour générer le secret et le QR code
            result = self.auth_manager.client.setup_2fa()

            self.secret = result.get("secret")
            qr_base64 = result.get("qr_code", "")

            # Afficher le secret
            if self.secret:
                # Formater le secret en groupes de 4
                formatted = " ".join([self.secret[i:i+4] for i in range(0, len(self.secret), 4)])
                self.secret_code.configure(text=formatted)

            # Décoder et afficher le QR code
            if qr_base64 and qr_base64.startswith("data:image"):
                # Extraire le base64 après la virgule
                base64_data = qr_base64.split(",")[1]
                image_data = base64.b64decode(base64_data)

                # Créer l'image
                img = Image.open(io.BytesIO(image_data))
                img = img.resize((180, 180))

                self.qr_image = ctk.CTkImage(light_image=img, dark_image=img, size=(180, 180))
                self.qr_label.configure(image=self.qr_image, text="")

        except Exception as e:
            self.error_label.configure(text=f"Erreur: {str(e)}")
            self.qr_label.configure(text="Erreur de chargement")

    def _verify(self):
        """Vérifie le code et active le 2FA"""
        code = self.code_entry.get().strip()

        if not code or len(code) != 6 or not code.isdigit():
            self.error_label.configure(text="Entrez un code à 6 chiffres")
            return

        try:
            self.verify_btn.configure(state="disabled", text="Vérification...")
            self.update()

            result = self.auth_manager.client.verify_2fa(code)

            if result.get("enabled"):
                messagebox.showinfo(
                    "2FA Activé",
                    "L'authentification à deux facteurs est maintenant activée.\n\n"
                    "Conservez votre app d'authentification en lieu sûr.\n"
                    "Vous aurez besoin du code à chaque connexion."
                )
                self.destroy()
            else:
                self.error_label.configure(text="Erreur lors de l'activation")
                self.verify_btn.configure(state="normal", text="Activer le 2FA")

        except Exception as e:
            error_msg = str(e)
            if "invalide" in error_msg.lower():
                self.error_label.configure(text="Code incorrect. Réessayez.")
            else:
                self.error_label.configure(text=f"Erreur: {error_msg}")
            self.verify_btn.configure(state="normal", text="Activer le 2FA")


class Disable2FADialog(ctk.CTkToplevel):
    """Dialogue pour désactiver le 2FA"""

    def __init__(self, parent, auth_manager):
        super().__init__(parent)

        self.auth_manager = auth_manager

        self.title("Désactiver 2FA")
        self.geometry("400x280")
        self.configure(fg_color="#0f1117")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._create_ui()

        # Centrer
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 280) // 2
        self.geometry(f"+{x}+{y}")

    def _create_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#401515", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="⚠️ Désactiver 2FA",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FF6B6B"
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkButton(
            header, text="✕", width=30, height=30,
            fg_color="transparent", hover_color="#2a2d36",
            command=self.destroy
        ).pack(side="right", padx=10, pady=10)

        # Content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=25, pady=20)

        ctk.CTkLabel(
            content,
            text="Pour désactiver le 2FA, entrez le code\naffiché dans votre app d'authentification:",
            font=ctk.CTkFont(size=13),
            text_color="#AAAAAA",
            justify="center"
        ).pack(pady=(0, 20))

        self.code_entry = ctk.CTkEntry(
            content,
            height=50,
            font=ctk.CTkFont(size=24, weight="bold", family="Courier"),
            fg_color="#1a1d24",
            border_color="#2a2d36",
            justify="center",
            placeholder_text="000000"
        )
        self.code_entry.pack(fill="x", pady=(0, 10))
        self.code_entry.bind("<Return>", lambda e: self._disable())

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
            height=40, corner_radius=8,
            command=self.destroy
        ).pack(side="left", expand=True, padx=(0, 5))

        self.disable_btn = ctk.CTkButton(
            btn_frame, text="Désactiver",
            fg_color="#FF4444", hover_color="#CC3333",
            text_color="#FFFFFF",
            height=40, corner_radius=8,
            command=self._disable
        )
        self.disable_btn.pack(side="right", expand=True, padx=(5, 0))

    def _disable(self):
        code = self.code_entry.get().strip()

        if not code or len(code) != 6 or not code.isdigit():
            self.error_label.configure(text="Entrez un code à 6 chiffres")
            return

        try:
            self.disable_btn.configure(state="disabled", text="...")
            self.update()

            result = self.auth_manager.client.disable_2fa(code)

            if not result.get("enabled"):
                messagebox.showinfo("2FA Désactivé", "L'authentification 2FA a été désactivée.")
                self.destroy()
            else:
                self.error_label.configure(text="Erreur lors de la désactivation")
                self.disable_btn.configure(state="normal", text="Désactiver")

        except Exception as e:
            error_msg = str(e)
            if "invalide" in error_msg.lower():
                self.error_label.configure(text="Code incorrect")
            else:
                self.error_label.configure(text=f"Erreur: {error_msg}")
            self.disable_btn.configure(state="normal", text="Désactiver")


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
            header, text="⚠️ SUPPRIMER LE COMPTE",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FF4444"
        ).pack(side="left", padx=20, pady=15)

        ctk.CTkButton(
            header, text="✕", width=35, height=35,
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
            text="🚨 ATTENTION - ACTION IRRÉVERSIBLE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FF6B6B"
        ).pack(padx=15, pady=(15, 5))

        ctk.CTkLabel(
            warning_frame,
            text="Cette action supprimera définitivement:\n• Votre compte et profil\n• Toutes vos données et préférences\n• Votre licence et abonnement",
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
            content, show="•", height=40,
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
                # Fermer l'application
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
