"""
HelixOne Academy - Interface de Formation Complète
Version 3.0 - Badges, Certifications et Gamification
"""

import customtkinter as ctk
from tkinter import messagebox
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.asset_path import get_base_path

# Définition des badges
BADGES = {
    "first_step": {"name": "Premiere pousse", "icon": "🌱", "condition": "first_module", "xp": 50},
    "studious": {"name": "Etudiant assidu", "icon": "📚", "condition": "7_days_streak", "xp": 100},
    "sharpshooter": {"name": "Tireur d'elite", "icon": "🎯", "condition": "quiz_100", "xp": 75},
    "diamond": {"name": "Diamant", "icon": "💎", "condition": "complete_parcours", "xp": 200},
    "champion": {"name": "Champion", "icon": "🏆", "condition": "all_parcours", "xp": 500},
    "mentor": {"name": "Mentor", "icon": "🤝", "condition": "help_10_users", "xp": 150},
    "on_fire": {"name": "En feu", "icon": "🔥", "condition": "30_days_streak", "xp": 300},
    "social": {"name": "Social", "icon": "💬", "condition": "50_messages", "xp": 100},
}

# Définition des certifications
CERTIFICATIONS = {
    "user": {"name": "HelixOne Certified User", "icon": "📜", "level": 1, "parcours": "debutant", "min_score": 80, "color": "#00D9FF"},
    "trader": {"name": "HelixOne Certified Trader", "icon": "📜", "level": 2, "parcours": "intermediaire", "min_score": 80, "color": "#FFA500"},
    "expert": {"name": "HelixOne Certified Expert", "icon": "📜", "level": 3, "parcours": "avance", "min_score": 85, "color": "#FF4444"},
    "master": {"name": "HelixOne Master", "icon": "🏅", "level": 4, "parcours": "expert", "min_score": 90, "color": "#FFD700"},
}

class FormationAcademy(ctk.CTkFrame):
    """Interface principale de formation HelixOne"""

    def __init__(self, parent):
        super().__init__(parent, fg_color="#0a0e12")
        self.pack(fill="both", expand=True)

        # Chargement des données
        self.modules_data = self.load_modules()
        self.user_progress = self.load_user_progress()
        self.current_view = "dashboard"
        self._changing_tab = False  # Flag pour eviter les appels recursifs
        self._last_tab = "Dashboard"  # Pour detecter les changements d'onglet

        # Construction de l'interface
        self.build_interface()

        # Afficher le dashboard par défaut
        self.show_dashboard()

    def load_modules(self):
        """Charge tous les modules depuis le JSON"""
        try:
            json_path = os.path.join(get_base_path(), "data", "formation_commerciale", "modules_complets.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                modules = json.load(f)

            # Organiser par parcours (4 niveaux)
            organized = {
                "debutant": [],
                "intermediaire": [],
                "avance": [],
                "expert": []
            }

            # Map pour normaliser les parcours (gérer les accents)
            parcours_map = {
                "débutant": "debutant",
                "debutant": "debutant",
                "intermédiaire": "intermediaire",
                "intermediaire": "intermediaire",
                "avancé": "avance",
                "avance": "avance",
                "expert": "expert"
            }

            for module in modules:
                parcours_original = module.get("parcours", "débutant")
                # Normaliser le parcours
                parcours = parcours_map.get(parcours_original, "debutant")

                if parcours in organized:
                    organized[parcours].append(module)

            total = sum(len(v) for v in organized.values())
            print(f"[✓] {total} modules chargés (Débutant: {len(organized['debutant'])}, Intermédiaire: {len(organized['intermediaire'])}, Avancé: {len(organized['avance'])}, Expert: {len(organized['expert'])})")
            return organized

        except Exception as e:
            print(f"[✗] Erreur chargement modules: {e}")
            import traceback
            traceback.print_exc()
            return {"debutant": [], "intermediaire": [], "avance": [], "expert": []}

    def load_user_progress(self):
        """Charge ou crée la progression utilisateur"""
        try:
            progress_path = os.path.join(get_base_path(), "data", "formation_commerciale", "user_progress.json")
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                # S'assurer que les nouveaux champs existent
                if "badges" not in progress:
                    progress["badges"] = []
                if "certifications" not in progress:
                    progress["certifications"] = []
                if "login_dates" not in progress:
                    progress["login_dates"] = []
                if "community_messages" not in progress:
                    progress["community_messages"] = 0
                if "users_helped" not in progress:
                    progress["users_helped"] = 0
                return progress
        except Exception:
            return {
                "completed_modules": [],
                "quiz_scores": {},
                "total_xp": 0,
                "level": 1,
                "current_parcours": "debutant",
                "badges": [],
                "certifications": [],
                "login_dates": [],
                "community_messages": 0,
                "users_helped": 0
            }

    def check_and_award_badges(self) -> List[str]:
        """Vérifie et attribue les badges mérités"""
        new_badges = []

        # Badge: Première pousse (premier module complété)
        if "first_step" not in self.user_progress["badges"]:
            if len(self.user_progress["completed_modules"]) >= 1:
                new_badges.append("first_step")

        # Badge: Tireur d'élite (100% à un quiz)
        if "sharpshooter" not in self.user_progress["badges"]:
            for score in self.user_progress["quiz_scores"].values():
                if score >= 100:
                    new_badges.append("sharpshooter")
                    break

        # Badge: Diamant (parcours complet)
        if "diamond" not in self.user_progress["badges"]:
            for parcours in ["debutant", "intermediaire", "avance", "expert"]:
                if self.is_parcours_complete(parcours):
                    new_badges.append("diamond")
                    break

        # Badge: Champion (tous les parcours)
        if "champion" not in self.user_progress["badges"]:
            all_complete = all(self.is_parcours_complete(p) for p in ["debutant", "intermediaire", "avance", "expert"])
            if all_complete:
                new_badges.append("champion")

        # Badge: Étudiant assidu (7 jours consécutifs)
        if "studious" not in self.user_progress["badges"]:
            if self.check_streak(7):
                new_badges.append("studious")

        # Badge: En feu (30 jours consécutifs)
        if "on_fire" not in self.user_progress["badges"]:
            if self.check_streak(30):
                new_badges.append("on_fire")

        # Badge: Social (50 messages communauté)
        if "social" not in self.user_progress["badges"]:
            if self.user_progress.get("community_messages", 0) >= 50:
                new_badges.append("social")

        # Badge: Mentor (10 utilisateurs aidés)
        if "mentor" not in self.user_progress["badges"]:
            if self.user_progress.get("users_helped", 0) >= 10:
                new_badges.append("mentor")

        # Attribuer les nouveaux badges
        for badge_id in new_badges:
            self.user_progress["badges"].append(badge_id)
            self.user_progress["total_xp"] += BADGES[badge_id]["xp"]

        if new_badges:
            self.save_user_progress()

        return new_badges

    def is_parcours_complete(self, parcours: str) -> bool:
        """Vérifie si un parcours est complètement terminé"""
        modules = self.modules_data.get(parcours, [])
        if not modules:
            return False
        for module in modules:
            if module.get("id") not in self.user_progress["completed_modules"]:
                return False
        return True

    def check_streak(self, days: int) -> bool:
        """Vérifie si l'utilisateur a une série de connexions consécutives"""
        login_dates = self.user_progress.get("login_dates", [])
        if len(login_dates) < days:
            return False

        # Convertir en dates et trier
        dates = sorted([datetime.fromisoformat(d).date() for d in login_dates], reverse=True)

        # Vérifier les jours consécutifs
        for i in range(days - 1):
            if i + 1 >= len(dates):
                return False
            diff = (dates[i] - dates[i + 1]).days
            if diff != 1:
                return False
        return True

    def record_login(self):
        """Enregistre la connexion du jour"""
        today = datetime.now().date().isoformat()
        if today not in self.user_progress.get("login_dates", []):
            if "login_dates" not in self.user_progress:
                self.user_progress["login_dates"] = []
            self.user_progress["login_dates"].append(today)
            # Garder seulement les 60 derniers jours
            self.user_progress["login_dates"] = self.user_progress["login_dates"][-60:]
            self.save_user_progress()

    def check_certification_eligibility(self, cert_id: str) -> tuple:
        """Vérifie si l'utilisateur est éligible à une certification"""
        cert = CERTIFICATIONS.get(cert_id)
        if not cert:
            return False, "Certification inconnue"

        # Vérifier si déjà obtenue
        if cert_id in self.user_progress.get("certifications", []):
            return False, "Certification déjà obtenue"

        # Vérifier le parcours
        parcours = cert["parcours"]
        if not self.is_parcours_complete(parcours):
            return False, f"Parcours {parcours} non complété"

        # Vérifier le score minimum aux quiz
        parcours_modules = self.modules_data.get(parcours, [])
        total_score = 0
        count = 0
        for module in parcours_modules:
            mod_id = module.get("id")
            if mod_id in self.user_progress["quiz_scores"]:
                total_score += self.user_progress["quiz_scores"][mod_id]
                count += 1

        if count == 0:
            return False, "Aucun quiz complété"

        avg_score = total_score / count
        if avg_score < cert["min_score"]:
            return False, f"Score moyen insuffisant ({avg_score:.0f}% < {cert['min_score']}%)"

        return True, "Éligible"

    def award_certification(self, cert_id: str) -> bool:
        """Attribue une certification"""
        eligible, reason = self.check_certification_eligibility(cert_id)
        if not eligible:
            return False

        if "certifications" not in self.user_progress:
            self.user_progress["certifications"] = []

        self.user_progress["certifications"].append(cert_id)
        self.user_progress["total_xp"] += 250  # Bonus XP pour certification
        self.save_user_progress()
        return True

    def save_user_progress(self):
        """Sauvegarde la progression utilisateur"""
        try:
            progress_path = os.path.join(get_base_path(), "data", "formation_commerciale", "user_progress.json")
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_progress, f, indent=2)
        except Exception as e:
            print(f"[✗] Erreur sauvegarde progression: {e}")

    def build_interface(self):
        """Construit l'interface principale avec navigation par onglets"""
        # Header compact avec stats
        self.header = self.create_header()
        self.header.pack(fill="x", padx=10, pady=(10, 5))

        # Navigation par onglets (remplace la sidebar)
        self.tabs = ctk.CTkTabview(
            self,
            fg_color="#161920",
            segmented_button_fg_color="#1c2028",
            segmented_button_selected_color="#00D9FF",
            segmented_button_selected_hover_color="#00B8E6",
            segmented_button_unselected_color="#1c2028",
            segmented_button_unselected_hover_color="#2a2d36",
            corner_radius=10
        )
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Créer les onglets
        self.tabs.add("Dashboard")
        self.tabs.add("Débutant")
        self.tabs.add("Intermédiaire")
        self.tabs.add("Avancé")
        self.tabs.add("Expert")

        # Configurer le callback APRES avoir cree les onglets
        self.tabs.configure(command=self._on_tab_change)

        # Zone de contenu pour chaque onglet (pour compatibilité)
        self.content_area = self.tabs.tab("Dashboard")

        # Liste pour compatibilité avec highlight_nav_button
        self.nav_buttons = []

        # Stocker le dernier onglet pour detecter les changements
        self._last_tab = "Dashboard"

        # Fallback: polling pour detecter les changements d'onglet
        # (au cas ou le callback ne fonctionne pas)
        self._start_tab_polling()

    def create_header(self):
        """Crée le header compact avec stats"""
        header = ctk.CTkFrame(self, height=50, fg_color="#161920", corner_radius=10)
        header.pack_propagate(False)

        # Titre compact
        title_label = ctk.CTkLabel(
            header,
            text="Academy",
            font=("Arial", 18, "bold"),
            text_color="#00D9FF"
        )
        title_label.pack(side="left", padx=20, pady=10)

        # Stats utilisateur (à droite)
        stats_frame = ctk.CTkFrame(header, fg_color="transparent")
        stats_frame.pack(side="right", padx=20, pady=10)

        # Badges count
        badges_count = len(self.user_progress.get("badges", []))
        ctk.CTkLabel(
            stats_frame,
            text=f"Badges: {badges_count}",
            font=("Arial", 12),
            text_color="#FF9500"
        ).pack(side="left", padx=10)

        # XP
        ctk.CTkLabel(
            stats_frame,
            text=f"{self.user_progress['total_xp']} XP",
            font=("Arial", 12, "bold"),
            text_color="#FFD700"
        ).pack(side="left", padx=10)

        # Level
        ctk.CTkLabel(
            stats_frame,
            text=f"Lv.{self.user_progress['level']}",
            font=("Arial", 13, "bold"),
            text_color="#00FF88"
        ).pack(side="left", padx=10)

        return header

    def _start_tab_polling(self):
        """Demarre le polling pour detecter les changements d'onglet"""
        def check_tab():
            try:
                if not self.winfo_exists():
                    return
                current_tab = self.tabs.get()
                if current_tab != self._last_tab:
                    print(f"[Formation] Polling detected tab change: {self._last_tab} -> {current_tab}")
                    self._last_tab = current_tab
                    self._on_tab_change(current_tab)
                # Verifier toutes les 200ms
                self.after(200, check_tab)
            except Exception as e:
                print(f"[Formation] Polling error: {e}")

        self.after(500, check_tab)

    def _on_tab_change(self, tab_name=None):
        """Gère le changement d'onglet"""
        # Recuperer le nom de l'onglet actif si non fourni
        if not tab_name:
            try:
                tab_name = self.tabs.get()
            except Exception:
                return

        print(f"[Formation] Tab change: '{tab_name}'")

        # Eviter les appels recursifs
        if hasattr(self, '_changing_tab') and self._changing_tab:
            return

        # Mettre a jour _last_tab AVANT de changer pour eviter boucle infinie avec polling
        self._last_tab = tab_name

        self._changing_tab = True
        try:
            tab_actions = {
                "Dashboard": self._load_dashboard_content,
                "Débutant": lambda: self._load_parcours_content("debutant"),
                "Intermédiaire": lambda: self._load_parcours_content("intermediaire"),
                "Avancé": lambda: self._load_parcours_content("avance"),
                "Expert": lambda: self._load_parcours_content("expert"),
            }
            action = tab_actions.get(tab_name)
            if action:
                action()
            else:
                print(f"[Formation] No action for tab: '{tab_name}'")
        except Exception as e:
            print(f"[Formation] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._changing_tab = False

    def _load_dashboard_content(self):
        """Charge le contenu du dashboard sans changer l'onglet"""
        self.content_area = self.tabs.tab("Dashboard")
        self.clear_content()
        self.current_view = "dashboard"
        self._build_dashboard_content()

    def _load_parcours_content(self, parcours_id):
        """Charge le contenu d'un parcours sans changer l'onglet"""
        tab_names = {
            "debutant": "Débutant",
            "intermediaire": "Intermédiaire",
            "avance": "Avancé",
            "expert": "Expert"
        }
        tab_name = tab_names.get(parcours_id, "Débutant")
        self.content_area = self.tabs.tab(tab_name)
        self.clear_content()
        self.current_view = f"parcours_{parcours_id}"
        self._build_parcours_content(parcours_id)

    def clear_content(self):
        """Nettoie la zone de contenu actuelle"""
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def highlight_nav_button(self, index):
        """Obsolète - conservé pour compatibilité"""
        pass

    def get_current_tab(self):
        """Retourne le conteneur de l'onglet actuel"""
        tab_name = self.tabs.get()
        return self.tabs.tab(tab_name)

    def show_dashboard(self):
        """Affiche le dashboard principal (avec changement d'onglet)"""
        self._changing_tab = True
        self._last_tab = "Dashboard"  # Mettre a jour pour eviter double appel
        try:
            self.tabs.set("Dashboard")
            self.content_area = self.tabs.tab("Dashboard")
            self.clear_content()
            self.current_view = "dashboard"
            self._build_dashboard_content()
        finally:
            self._changing_tab = False

    def _build_dashboard_content(self):
        """Construit le contenu du dashboard"""
        # Enregistrer la connexion du jour
        self.record_login()

        # Vérifier les nouveaux badges
        new_badges = self.check_and_award_badges()
        if new_badges:
            self.show_badge_notification(new_badges)

        # Scroll frame
        scroll = ctk.CTkScrollableFrame(
            self.content_area,
            fg_color="transparent"
        )
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        title = ctk.CTkLabel(
            scroll,
            text="Votre Progression",
            font=("Arial", 28, "bold"),
            text_color="#00D9FF"
        )
        title.pack(pady=(0, 30))

        # Stats cards
        stats_container = ctk.CTkFrame(scroll, fg_color="transparent")
        stats_container.pack(fill="x", pady=20)

        # Calculer les stats
        total_modules = sum(len(modules) for modules in self.modules_data.values())
        completed = len(self.user_progress['completed_modules'])
        progress_pct = int((completed / total_modules * 100)) if total_modules > 0 else 0
        badges_count = len(self.user_progress.get('badges', []))
        certs_count = len(self.user_progress.get('certifications', []))

        stats = [
            ("Modules Complétés", f"{completed}/{total_modules}", "#00FF88"),
            ("Progression Globale", f"{progress_pct}%", "#00D9FF"),
            ("XP Total", f"{self.user_progress['total_xp']}", "#FFD700"),
            ("Badges", f"{badges_count}/8", "#FF6B6B"),
            ("Certifications", f"{certs_count}/4", "#9B59B6")
        ]

        for i, (label, value, color) in enumerate(stats):
            card = self.create_stat_card(stats_container, label, value, color)
            card.grid(row=0, column=i, padx=10, sticky="ew")
            stats_container.grid_columnconfigure(i, weight=1)

        # Section Badges
        self.create_badges_section(scroll)

        # Section Certifications
        self.create_certifications_section(scroll)

        # Parcours disponibles
        parcours_title = ctk.CTkLabel(
            scroll,
            text="Parcours de Formation",
            font=("Arial", 22, "bold"),
            text_color="#FFFFFF"
        )
        parcours_title.pack(pady=(30, 20))

        # Cards des parcours (4 niveaux)
        for parcours_id, parcours_name, color, level_num, locked in [
            ("debutant", "Trader Débutant", "#00D9FF", "1", False),
            ("intermediaire", "Trader Confirmé", "#FFA500", "2", False),
            ("avance", "Trader Avancé", "#FF4444", "3", not self.is_parcours_complete("intermediaire")),
            ("expert", "Trader Expert", "#9B59B6", "4", not self.is_parcours_complete("avance"))
        ]:
            self.create_parcours_card(scroll, parcours_id, parcours_name, color, level_num, locked)

    def create_badges_section(self, parent):
        """Crée la section des badges"""
        section = ctk.CTkFrame(parent, fg_color="#1c2028", corner_radius=15)
        section.pack(fill="x", pady=20)

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="Badges Obtenus",
            font=("Arial", 20, "bold"),
            text_color="#FFD700"
        ).pack(side="left")

        user_badges = self.user_progress.get("badges", [])
        ctk.CTkLabel(
            header,
            text=f"{len(user_badges)}/8",
            font=("Arial", 14),
            text_color="#888888"
        ).pack(side="right")

        # Grille des badges
        badges_frame = ctk.CTkFrame(section, fg_color="transparent")
        badges_frame.pack(fill="x", padx=20, pady=(0, 15))

        for i, (badge_id, badge_data) in enumerate(BADGES.items()):
            is_earned = badge_id in user_badges
            self.create_badge_widget(badges_frame, badge_id, badge_data, is_earned, i)

    def create_badge_widget(self, parent, badge_id, badge_data, is_earned, index):
        """Crée un widget de badge"""
        frame = ctk.CTkFrame(
            parent,
            fg_color="#2a2d36" if is_earned else "#16181c",
            corner_radius=10,
            width=80,
            height=90
        )
        frame.grid(row=0, column=index, padx=5, pady=5)
        frame.grid_propagate(False)

        # Icône
        icon_color = "#FFFFFF" if is_earned else "#444444"
        ctk.CTkLabel(
            frame,
            text=badge_data["icon"],
            font=("Arial", 28),
            text_color=icon_color
        ).pack(pady=(10, 0))

        # Nom
        name_color = "#FFFFFF" if is_earned else "#555555"
        ctk.CTkLabel(
            frame,
            text=badge_data["name"][:10],
            font=("Arial", 9),
            text_color=name_color
        ).pack(pady=(5, 0))

        # XP si gagné
        if is_earned:
            ctk.CTkLabel(
                frame,
                text=f"+{badge_data['xp']} XP",
                font=("Arial", 8),
                text_color="#FFD700"
            ).pack()

    def create_certifications_section(self, parent):
        """Crée la section des certifications"""
        section = ctk.CTkFrame(parent, fg_color="#1c2028", corner_radius=15)
        section.pack(fill="x", pady=10)

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="Certifications",
            font=("Arial", 20, "bold"),
            text_color="#9B59B6"
        ).pack(side="left")

        user_certs = self.user_progress.get("certifications", [])
        ctk.CTkLabel(
            header,
            text=f"{len(user_certs)}/4",
            font=("Arial", 14),
            text_color="#888888"
        ).pack(side="right")

        # Liste des certifications
        certs_frame = ctk.CTkFrame(section, fg_color="transparent")
        certs_frame.pack(fill="x", padx=20, pady=(0, 15))

        for cert_id, cert_data in CERTIFICATIONS.items():
            is_earned = cert_id in user_certs
            eligible, reason = self.check_certification_eligibility(cert_id)
            self.create_certification_widget(certs_frame, cert_id, cert_data, is_earned, eligible, reason)

    def create_certification_widget(self, parent, cert_id, cert_data, is_earned, eligible, reason):
        """Crée un widget de certification"""
        frame = ctk.CTkFrame(
            parent,
            fg_color="#2a2d36" if is_earned else "#16181c",
            corner_radius=10
        )
        frame.pack(fill="x", pady=5)

        content = ctk.CTkFrame(frame, fg_color="transparent")
        content.pack(fill="x", padx=15, pady=10)

        # Icône et nom
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.pack(side="left")

        icon_text = cert_data["icon"] if is_earned else "🔒"
        ctk.CTkLabel(
            left,
            text=icon_text,
            font=("Arial", 24)
        ).pack(side="left", padx=(0, 10))

        name_color = cert_data["color"] if is_earned else "#666666"
        ctk.CTkLabel(
            left,
            text=cert_data["name"],
            font=("Arial", 14, "bold"),
            text_color=name_color
        ).pack(side="left")

        # Statut
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.pack(side="right")

        if is_earned:
            ctk.CTkLabel(
                right,
                text="✓ Obtenue",
                font=("Arial", 12),
                text_color="#00FF88"
            ).pack()
        elif eligible:
            btn = ctk.CTkButton(
                right,
                text="Obtenir",
                font=("Arial", 12),
                fg_color=cert_data["color"],
                width=80,
                height=30,
                command=lambda c=cert_id: self.claim_certification(c)
            )
            btn.pack()
        else:
            ctk.CTkLabel(
                right,
                text=reason,
                font=("Arial", 10),
                text_color="#888888"
            ).pack()

    def claim_certification(self, cert_id):
        """Réclame une certification"""
        if self.award_certification(cert_id):
            cert = CERTIFICATIONS[cert_id]
            messagebox.showinfo(
                "Certification Obtenue!",
                f"Félicitations! Vous avez obtenu la certification:\n\n{cert['icon']} {cert['name']}\n\n+250 XP"
            )
            self.show_dashboard()
        else:
            messagebox.showerror("Erreur", "Impossible d'obtenir cette certification.")

    def show_badge_notification(self, badge_ids):
        """Affiche une notification pour les nouveaux badges"""
        badge_names = [f"{BADGES[b]['icon']} {BADGES[b]['name']}" for b in badge_ids]
        messagebox.showinfo(
            "Nouveau Badge!",
            f"Félicitations! Vous avez obtenu:\n\n" + "\n".join(badge_names)
        )

    def create_stat_card(self, parent, label, value, color):
        """Crée une carte de statistique"""
        card = ctk.CTkFrame(parent, fg_color="#1c2028", corner_radius=10)

        label_widget = ctk.CTkLabel(
            card,
            text=label,
            font=("Arial", 12),
            text_color="#888888"
        )
        label_widget.pack(pady=(15, 5))

        value_widget = ctk.CTkLabel(
            card,
            text=value,
            font=("Arial", 28, "bold"),
            text_color=color
        )
        value_widget.pack(pady=(0, 15))

        return card

    def create_parcours_card(self, parent, parcours_id, parcours_name, color, level_num, locked=False):
        """Crée une carte de parcours avec bordure distinctive"""
        modules = self.modules_data.get(parcours_id, [])
        completed = len([m for m in modules if m.get('id') in self.user_progress['completed_modules']])
        total = len(modules)
        progress = int((completed / total * 100)) if total > 0 else 0
        is_complete = completed == total and total > 0

        # Bordure selon l'état: vert si complet, couleur de niveau si en cours, gris si verrouillé
        if locked:
            border_color = "#333333"
            bg_color = "#12151a"
        elif is_complete:
            border_color = "#00FF88"
            bg_color = "#1c2028"
        else:
            border_color = color
            bg_color = "#1c2028"

        card = ctk.CTkFrame(
            parent,
            fg_color=bg_color,
            corner_radius=12,
            border_width=2,
            border_color=border_color
        )
        card.pack(fill="x", pady=8)

        # Header
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(16, 10))

        # Badge niveau
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left")

        # Utiliser des couleurs sombres pour le fond du badge (pas d'alpha en Tkinter)
        badge_colors = {
            "#00D9FF": "#1a3a4a",  # Cyan foncé
            "#FFA500": "#4a3a1a",  # Orange foncé
            "#FF4444": "#4a1a1a",  # Rouge foncé
            "#9B59B6": "#3a1a4a",  # Violet foncé
        }
        badge_bg = badge_colors.get(color, "#1a1a2a") if not locked else "#1a1a1a"
        badge_fg = color if not locked else "#555555"
        level_badge = ctk.CTkFrame(
            title_frame,
            fg_color=badge_bg,
            corner_radius=6,
            width=36,
            height=36
        )
        level_badge.pack(side="left", padx=(0, 12))
        level_badge.pack_propagate(False)

        level_label = ctk.CTkLabel(
            level_badge,
            text=level_num,
            font=("Arial", 16, "bold"),
            text_color=badge_fg
        )
        level_label.place(relx=0.5, rely=0.5, anchor="center")

        title_color = color if not locked else "#555555"
        title_text = parcours_name if not locked else f"{parcours_name} (Verrouillé)"
        title = ctk.CTkLabel(
            title_frame,
            text=title_text,
            font=("Arial", 18, "bold"),
            text_color=title_color
        )
        title.pack(side="left")

        status_text = f"{completed}/{total} modules" if not locked else "Verrouillé"
        status = ctk.CTkLabel(
            header_frame,
            text=status_text,
            font=("Arial", 14),
            text_color="#888888" if not locked else "#555555"
        )
        status.pack(side="right")

        # Barre de progression
        progress_frame = ctk.CTkFrame(card, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=(0, 10))

        progress_bar = ctk.CTkProgressBar(
            progress_frame,
            height=10,
            progress_color=color if not locked else "#333333"
        )
        progress_bar.pack(fill="x")
        progress_bar.set(progress / 100 if not locked else 0)

        if locked:
            progress_label = ctk.CTkLabel(
                progress_frame,
                text="Complétez le parcours précédent pour débloquer",
                font=("Arial", 11),
                text_color="#666666"
            )
        else:
            progress_label = ctk.CTkLabel(
                progress_frame,
                text=f"{progress}% complété",
                font=("Arial", 12),
                text_color="#CCCCCC"
            )
        progress_label.pack(pady=(5, 0))

        # Bouton d'action
        if locked:
            btn = ctk.CTkButton(
                card,
                text="Verrouillé",
                font=("Arial", 13, "bold"),
                fg_color="#333333",
                hover_color="#333333",
                height=38,
                corner_radius=8,
                state="disabled"
            )
        else:
            btn = ctk.CTkButton(
                card,
                text="Commencer" if completed == 0 else "Continuer",
                font=("Arial", 14, "bold"),
                fg_color=color,
                hover_color=color,
                height=40,
                command=lambda: self.show_parcours(parcours_id)
            )
        btn.pack(pady=(10, 20), padx=20)

    def show_parcours(self, parcours_id):
        """Affiche les modules d'un parcours (avec changement d'onglet)"""
        self._changing_tab = True
        tab_names = {
            "debutant": "Débutant",
            "intermediaire": "Intermédiaire",
            "avance": "Avancé",
            "expert": "Expert"
        }
        tab_name = tab_names.get(parcours_id, "Débutant")
        self._last_tab = tab_name  # Mettre a jour pour eviter double appel
        try:
            self.tabs.set(tab_name)
            self.content_area = self.tabs.tab(tab_name)
            self.clear_content()
            self.current_view = f"parcours_{parcours_id}"
            self._build_parcours_content(parcours_id)
        finally:
            self._changing_tab = False

    def _build_parcours_content(self, parcours_id):
        """Construit le contenu d'un parcours"""
        modules = self.modules_data.get(parcours_id, [])

        if not modules:
            empty_label = ctk.CTkLabel(
                self.content_area,
                text="Aucun module disponible pour ce parcours",
                font=("Helvetica", 16),
                text_color="#6B7280"
            )
            empty_label.pack(expand=True)
            return

        # Scroll frame
        scroll = ctk.CTkScrollableFrame(
            self.content_area,
            fg_color="transparent"
        )
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre du parcours
        parcours_info = {
            "debutant": ("Formation Débutant", "#00D9FF", "Apprenez les bases du trading et des marchés financiers"),
            "intermediaire": ("Formation Intermédiaire", "#FFA500", "Approfondissez vos connaissances avec l'analyse technique"),
            "avance": ("Formation Avancé", "#FF6B6B", "Maîtrisez les stratégies avancées de trading"),
            "expert": ("Formation Expert", "#C084FC", "Devenez un trader professionnel")
        }

        name, color, subtitle = parcours_info.get(parcours_id, ("Formation", "#00D9FF", ""))

        # Compteur progression
        completed = sum(1 for m in modules if m.get('id', '') in self.user_progress['completed_modules'])
        total = len(modules)

        title_area = ctk.CTkFrame(scroll, fg_color="transparent")
        title_area.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            title_area, text=name,
            font=("Helvetica", 28, "bold"), text_color="#FFFFFF"
        ).pack(anchor="w")

        if subtitle:
            ctk.CTkLabel(
                title_area, text=subtitle,
                font=("Helvetica", 14), text_color="#8B949E"
            ).pack(anchor="w", pady=(4, 0))

        # Barre de progression du parcours
        progress_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        progress_frame.pack(fill="x", pady=(10, 20))

        ctk.CTkLabel(
            progress_frame, text=f"{completed}/{total} modules complétés",
            font=("Helvetica", 12), text_color="#8B949E"
        ).pack(anchor="w", pady=(0, 6))

        bar_bg = ctk.CTkFrame(progress_frame, fg_color="#21262d", height=6, corner_radius=3)
        bar_bg.pack(fill="x")

        if total > 0:
            ratio = completed / total
            bar_fg = ctk.CTkFrame(bar_bg, fg_color=color, height=6, corner_radius=3)
            bar_fg.place(relx=0, rely=0, relwidth=max(ratio, 0.0), relheight=1.0)

        # Afficher chaque module
        for i, module in enumerate(modules, 1):
            self.create_module_card(scroll, module, i)

    def create_module_card(self, parent, module, index):
        """Crée une carte de module avec design soigné"""
        module_id = module.get('id', '')
        is_completed = module_id in self.user_progress['completed_modules']

        # Quiz score si disponible
        quiz_score = self.user_progress.get('quiz_scores', {}).get(module_id, None)

        card = ctk.CTkFrame(
            parent,
            fg_color="#161b22",
            corner_radius=12,
            border_width=1,
            border_color="#238636" if is_completed else "#1e2734"
        )
        card.pack(fill="x", pady=6)

        # Container principal
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=20, pady=18)

        # Numéro et statut
        left_frame = ctk.CTkFrame(content_frame, fg_color="transparent", width=52)
        left_frame.pack(side="left", anchor="n")
        left_frame.pack_propagate(False)

        number_bg = "#1a3a2a" if is_completed else "#21262d"
        number_fg = "#3fb950" if is_completed else "#8B949E"

        number_badge = ctk.CTkFrame(
            left_frame, fg_color=number_bg, corner_radius=26,
            width=46, height=46
        )
        number_badge.pack(pady=(0, 4))
        number_badge.pack_propagate(False)

        ctk.CTkLabel(
            number_badge, text=str(index),
            font=("Helvetica", 18, "bold"), text_color=number_fg
        ).place(relx=0.5, rely=0.5, anchor="center")

        if is_completed:
            ctk.CTkLabel(
                left_frame, text="Fait",
                font=("Helvetica", 10), text_color="#3fb950"
            ).pack()

        # Détails du module
        details_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        details_frame.pack(side="left", fill="both", expand=True, padx=(16, 0))

        ctk.CTkLabel(
            details_frame,
            text=module.get('titre', 'Module sans titre'),
            font=("Helvetica", 17, "bold"),
            text_color="#FFFFFF",
            anchor="w"
        ).pack(fill="x")

        desc = module.get('description', '')
        if desc:
            ctk.CTkLabel(
                details_frame, text=desc,
                font=("Helvetica", 13), text_color="#8B949E",
                anchor="w", wraplength=520, justify="left"
            ).pack(fill="x", pady=(4, 10))

        # Tags d'info (durée, difficulté, XP)
        info_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        info_frame.pack(fill="x")

        duree = module.get('duree', '') or module.get('durée', '')
        difficulte = module.get('difficulte', '') or module.get('difficulté', '')
        xp = module.get('points_xp', 0)

        tags = []
        if duree:
            tags.append(duree)
        if difficulte:
            tags.append(difficulte)
        if xp:
            tags.append(f"{xp} XP")
        if quiz_score is not None:
            tags.append(f"Quiz: {quiz_score}%")

        for tag_text in tags:
            tag = ctk.CTkFrame(info_frame, fg_color="#21262d", corner_radius=6)
            tag.pack(side="left", padx=(0, 8))

            ctk.CTkLabel(
                tag, text=tag_text,
                font=("Helvetica", 11), text_color="#8B949E"
            ).pack(padx=10, pady=4)

        # Bouton d'action
        if is_completed:
            btn_text, btn_fg, btn_hover = "Revoir", "#21262d", "#30363d"
            btn_text_color = "#8B949E"
        else:
            btn_text, btn_fg, btn_hover = "Commencer", "#238636", "#2ea043"
            btn_text_color = "#FFFFFF"

        ctk.CTkButton(
            content_frame, text=btn_text,
            font=("Helvetica", 13, "bold"),
            fg_color=btn_fg, hover_color=btn_hover,
            text_color=btn_text_color,
            width=110, height=38, corner_radius=8,
            command=lambda: self.open_module(module)
        ).pack(side="right", anchor="n")

    def open_module(self, module):
        """Ouvre un module pour le visualiser"""
        self.clear_content()

        # Créer le viewer de module
        viewer = ModuleViewer(self.content_area, module, self)
        viewer.pack(fill="both", expand=True)

    def mark_module_completed(self, module_id, xp_earned):
        """Marque un module comme complété"""
        if module_id not in self.user_progress['completed_modules']:
            self.user_progress['completed_modules'].append(module_id)
            self.user_progress['total_xp'] += xp_earned

            # Calculer le nouveau niveau (1 niveau tous les 500 XP)
            self.user_progress['level'] = (self.user_progress['total_xp'] // 500) + 1

            self.save_user_progress()

            # Mettre à jour le header sans reconstruire toute l'interface
            self._refresh_header()

            messagebox.showinfo(
                "Module Complété!",
                f"Félicitations! Vous avez gagné {xp_earned} XP!\n\nTotal XP: {self.user_progress['total_xp']}"
            )

    def _refresh_header(self):
        """Met à jour le header avec les stats actuelles sans reconstruire l'interface"""
        try:
            self.header.destroy()
            self.header = self.create_header()
            self.header.pack(fill="x", padx=10, pady=(10, 5), before=self.tabs)
        except Exception:
            pass

class ModuleViewer(ctk.CTkFrame):
    """Visualiseur de module complet"""

    def __init__(self, parent, module, academy):
        super().__init__(parent, fg_color="transparent")

        self.module = module
        self.academy = academy

        # Header avec retour
        header = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=12)
        header.pack(fill="x", padx=20, pady=(20, 10))

        header_inner = ctk.CTkFrame(header, fg_color="transparent")
        header_inner.pack(fill="x", padx=20, pady=16)

        back_btn = ctk.CTkButton(
            header_inner,
            text="← Retour",
            font=("Helvetica", 13),
            fg_color="#21262d",
            hover_color="#30363d",
            text_color="#8B949E",
            width=90,
            height=34,
            corner_radius=8,
            command=self.go_back
        )
        back_btn.pack(side="left", padx=(0, 18))

        # Titre du module
        title_frame = ctk.CTkFrame(header_inner, fg_color="transparent")
        title_frame.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_frame,
            text=module.get('titre', 'Module'),
            font=("Helvetica", 24, "bold"),
            text_color="#FFFFFF",
            anchor="w"
        ).pack(anchor="w")

        # Sous-titre avec description courte
        desc = module.get('description', '')
        if desc:
            ctk.CTkLabel(
                title_frame,
                text=desc[:120] + ('...' if len(desc) > 120 else ''),
                font=("Helvetica", 13),
                text_color="#8B949E",
                anchor="w"
            ).pack(anchor="w", pady=(4, 0))

        # Zone de contenu scrollable
        self.scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)

        # Afficher le contenu du module
        self.display_module_content()

        # Footer avec actions
        self.create_footer()

    def display_module_content(self):
        """Affiche le contenu complet du module avec une présentation soignée"""
        contenu = self.module.get('contenu', {})
        sections = contenu.get('sections', [])
        total_sections = len(sections)

        # --- Bandeau d'info du module ---
        info_bar = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=12)
        info_bar.pack(fill="x", pady=(0, 15))

        info_inner = ctk.CTkFrame(info_bar, fg_color="transparent")
        info_inner.pack(fill="x", padx=25, pady=16)

        # Difficulté, durée, XP
        parcours = self.module.get('parcours', '').capitalize()
        duree = self.module.get('duree', '')
        xp = self.module.get('points_xp', 0)
        difficulte = self.module.get('difficulte', '')

        parcours_colors = {
            'Debutant': '#00D9FF', 'Intermediaire': '#FFA500',
            'Avance': '#FF6B6B', 'Expert': '#C084FC'
        }
        accent = parcours_colors.get(parcours, '#00D9FF')

        meta_items = []
        if difficulte:
            meta_items.append(f"Niveau : {difficulte}")
        if duree:
            meta_items.append(f"Durée : {duree}")
        if xp:
            meta_items.append(f"Récompense : {xp} XP")
        if total_sections:
            meta_items.append(f"{total_sections} sections")

        meta_text = "   |   ".join(meta_items)
        ctk.CTkLabel(
            info_inner, text=meta_text,
            font=("Helvetica", 13), text_color="#8899AA"
        ).pack(anchor="w")

        # Barre de progression visuelle (accent colorée)
        bar_bg = ctk.CTkFrame(info_bar, fg_color="#0d1117", height=4, corner_radius=2)
        bar_bg.pack(fill="x", padx=25, pady=(0, 14))
        bar_fg = ctk.CTkFrame(bar_bg, fg_color=accent, height=4, corner_radius=2)
        bar_fg.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)

        # --- Introduction ---
        intro = contenu.get('introduction', '')
        if intro:
            intro_frame = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=12)
            intro_frame.pack(fill="x", pady=(0, 20))

            # Barre accent gauche simulée via un petit frame coloré
            intro_header = ctk.CTkFrame(intro_frame, fg_color="transparent")
            intro_header.pack(fill="x", padx=25, pady=(22, 0))

            ctk.CTkFrame(
                intro_header, fg_color=accent, width=4, height=22, corner_radius=2
            ).pack(side="left", padx=(0, 12))

            ctk.CTkLabel(
                intro_header, text="Introduction",
                font=("Helvetica", 20, "bold"), text_color="#FFFFFF"
            ).pack(side="left")

            ctk.CTkLabel(
                intro_frame, text=intro,
                font=("Helvetica", 14), text_color="#C8D1DA",
                wraplength=720, justify="left", anchor="nw"
            ).pack(fill="x", padx=25, pady=(14, 24))

        # --- Sections ---
        for i, section in enumerate(sections, 1):
            self._render_section(section, i, total_sections, accent)

        # --- Résumé ---
        resume = contenu.get('resume', '')
        if resume:
            resume_frame = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=12)
            resume_frame.pack(fill="x", pady=(10, 20))

            resume_header = ctk.CTkFrame(resume_frame, fg_color="transparent")
            resume_header.pack(fill="x", padx=25, pady=(22, 0))

            ctk.CTkFrame(
                resume_header, fg_color="#00D9FF", width=4, height=22, corner_radius=2
            ).pack(side="left", padx=(0, 12))

            ctk.CTkLabel(
                resume_header, text="Résumé",
                font=("Helvetica", 20, "bold"), text_color="#FFFFFF"
            ).pack(side="left")

            # Encadré résumé avec fond légèrement distinct
            resume_box = ctk.CTkFrame(resume_frame, fg_color="#0d1117", corner_radius=10)
            resume_box.pack(fill="x", padx=25, pady=(14, 24))

            ctk.CTkLabel(
                resume_box, text=resume,
                font=("Helvetica", 14), text_color="#C8D1DA",
                wraplength=690, justify="left", anchor="nw"
            ).pack(fill="x", padx=18, pady=16)

        # --- Ressources complémentaires ---
        ressources = contenu.get('ressources_complementaires', [])
        if ressources:
            res_frame = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=12)
            res_frame.pack(fill="x", pady=(0, 20))

            res_header = ctk.CTkFrame(res_frame, fg_color="transparent")
            res_header.pack(fill="x", padx=25, pady=(22, 0))

            ctk.CTkFrame(
                res_header, fg_color="#FFA500", width=4, height=22, corner_radius=2
            ).pack(side="left", padx=(0, 12))

            ctk.CTkLabel(
                res_header, text="Ressources Complémentaires",
                font=("Helvetica", 18, "bold"), text_color="#FFFFFF"
            ).pack(side="left")

            for ressource in ressources:
                row = ctk.CTkFrame(res_frame, fg_color="transparent")
                row.pack(fill="x", padx=35, pady=4)

                ctk.CTkLabel(
                    row, text="→", font=("Helvetica", 14, "bold"),
                    text_color="#FFA500", width=20
                ).pack(side="left")

                ctk.CTkLabel(
                    row, text=ressource,
                    font=("Helvetica", 13), text_color="#AAB8C8",
                    anchor="w", wraplength=680, justify="left"
                ).pack(side="left", padx=(8, 0))

            # Espacement bas
            ctk.CTkFrame(res_frame, fg_color="transparent", height=16).pack()

        # Section Q&A
        self.create_qa_section()

    def _render_section(self, section, index, total, accent):
        """Rendu d'une section de cours avec mise en page soignée"""
        section_frame = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=12)
        section_frame.pack(fill="x", pady=(0, 18))

        # --- En-tête de section ---
        header = ctk.CTkFrame(section_frame, fg_color="transparent")
        header.pack(fill="x", padx=25, pady=(22, 0))

        # Numéro dans un cercle coloré
        num_badge = ctk.CTkFrame(header, fg_color=accent, width=34, height=34, corner_radius=17)
        num_badge.pack(side="left", padx=(0, 14))
        num_badge.pack_propagate(False)

        ctk.CTkLabel(
            num_badge, text=str(index),
            font=("Helvetica", 15, "bold"), text_color="#FFFFFF"
        ).place(relx=0.5, rely=0.5, anchor="center")

        # Titre de section
        ctk.CTkLabel(
            header, text=section.get('titre', 'Section'),
            font=("Helvetica", 19, "bold"), text_color="#FFFFFF",
            anchor="w"
        ).pack(side="left", fill="x", expand=True)

        # Indicateur léger (section X sur Y)
        ctk.CTkLabel(
            header, text=f"{index}/{total}",
            font=("Helvetica", 12), text_color="#556677"
        ).pack(side="right")

        # Trait séparateur sous le titre
        sep = ctk.CTkFrame(section_frame, fg_color="#1e2734", height=1)
        sep.pack(fill="x", padx=25, pady=(14, 0))

        # --- Contenu principal ---
        section_content = section.get('contenu', '')
        if section_content:
            # Découper le contenu en paragraphes pour une meilleure lisibilité
            paragraphs = [p.strip() for p in section_content.split('\n') if p.strip()]

            content_area = ctk.CTkFrame(section_frame, fg_color="transparent")
            content_area.pack(fill="x", padx=25, pady=(16, 0))

            for paragraph in paragraphs:
                ctk.CTkLabel(
                    content_area, text=paragraph,
                    font=("Helvetica", 14), text_color="#C8D1DA",
                    wraplength=720, justify="left", anchor="nw"
                ).pack(fill="x", anchor="w", pady=(0, 10))

        # --- Exemple pratique ---
        exemple = section.get('exemple_pratique', '')
        if exemple:
            ex_frame = ctk.CTkFrame(section_frame, fg_color="#0d1117", corner_radius=10,
                                     border_width=1, border_color="#1e2734")
            ex_frame.pack(fill="x", padx=25, pady=(12, 0))

            ex_header = ctk.CTkFrame(ex_frame, fg_color="transparent")
            ex_header.pack(fill="x", padx=16, pady=(14, 0))

            ctk.CTkLabel(
                ex_header, text="Exemple pratique",
                font=("Helvetica", 14, "bold"), text_color="#FFC857"
            ).pack(side="left")

            ctk.CTkLabel(
                ex_frame, text=exemple,
                font=("Helvetica", 13), text_color="#B0BFCF",
                wraplength=680, justify="left", anchor="nw"
            ).pack(fill="x", padx=16, pady=(10, 16))

        # --- Points clés ---
        points_cles = section.get('points_cles', [])
        if points_cles:
            pc_frame = ctk.CTkFrame(section_frame, fg_color="#0f1923", corner_radius=10)
            pc_frame.pack(fill="x", padx=25, pady=(14, 0))

            pc_header = ctk.CTkFrame(pc_frame, fg_color="transparent")
            pc_header.pack(fill="x", padx=16, pady=(14, 8))

            ctk.CTkLabel(
                pc_header, text="Points Clés à Retenir",
                font=("Helvetica", 15, "bold"), text_color="#00E68C"
            ).pack(side="left")

            for point in points_cles:
                point_row = ctk.CTkFrame(pc_frame, fg_color="transparent")
                point_row.pack(fill="x", padx=16, pady=3)

                ctk.CTkLabel(
                    point_row, text="●",
                    font=("Helvetica", 8), text_color="#00E68C", width=16
                ).pack(side="left", anchor="n", pady=(5, 0))

                ctk.CTkLabel(
                    point_row, text=point,
                    font=("Helvetica", 13), text_color="#D0D8E0",
                    wraplength=670, justify="left", anchor="nw"
                ).pack(side="left", fill="x", padx=(6, 0))

            # Espacement bas
            ctk.CTkFrame(pc_frame, fg_color="transparent", height=12).pack()

        # Espacement final de la section
        ctk.CTkFrame(section_frame, fg_color="transparent", height=8).pack()

    def create_qa_section(self):
        """Crée la section Questions & Réponses du module"""
        qa_frame = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=12)
        qa_frame.pack(fill="x", pady=(10, 0))

        # Titre
        header = ctk.CTkFrame(qa_frame, fg_color="transparent")
        header.pack(fill="x", padx=25, pady=(22, 10))

        qa_header_left = ctk.CTkFrame(header, fg_color="transparent")
        qa_header_left.pack(side="left")

        ctk.CTkFrame(
            qa_header_left, fg_color="#58A6FF", width=4, height=22, corner_radius=2
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            qa_header_left,
            text="Questions & Réponses",
            font=("Helvetica", 18, "bold"),
            text_color="#FFFFFF"
        ).pack(side="left")

        # Bouton poser une question
        ctk.CTkButton(
            header,
            text="+ Poser une Question",
            width=160,
            height=32,
            font=("Helvetica", 12),
            fg_color="#21262d",
            hover_color="#30363d",
            text_color="#58A6FF",
            corner_radius=8,
            command=self.open_ask_question_dialog
        ).pack(side="right")

        # Liste des Q&A
        qa_list = ctk.CTkFrame(qa_frame, fg_color="transparent")
        qa_list.pack(fill="x", padx=25, pady=(0, 22))

        # Charger les questions pour ce module
        module_id = self.module.get('id', '')
        questions = self.load_module_questions(module_id)

        if not questions:
            ctk.CTkLabel(
                qa_list,
                text="Aucune question pour ce module.\nSoyez le premier à poser une question!",
                font=("Arial", 12),
                text_color="#888888",
                justify="center"
            ).pack(pady=20)
        else:
            for q in questions[:5]:  # Afficher max 5 questions
                self.create_qa_card(qa_list, q)

            if len(questions) > 5:
                ctk.CTkButton(
                    qa_list,
                    text=f"Voir toutes les questions ({len(questions)})",
                    fg_color="transparent",
                    text_color="#00D9FF",
                    hover_color="#1c2028",
                    command=lambda: self.show_all_questions(module_id)
                ).pack(pady=10)

    def create_qa_card(self, parent, question: dict):
        """Crée une carte de question"""
        card = ctk.CTkFrame(parent, fg_color="#0d1117", corner_radius=10,
                            border_width=1, border_color="#1e2734")
        card.pack(fill="x", pady=5)

        # Question
        q_frame = ctk.CTkFrame(card, fg_color="transparent")
        q_frame.pack(fill="x", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            q_frame,
            text=f"Q: {question.get('question', '')}",
            font=("Arial", 12, "bold"),
            text_color="#FFFFFF",
            wraplength=650,
            justify="left",
            anchor="w"
        ).pack(anchor="w")

        # Métadonnées (auteur, date)
        meta_frame = ctk.CTkFrame(card, fg_color="transparent")
        meta_frame.pack(fill="x", padx=15)

        ctk.CTkLabel(
            meta_frame,
            text=f"Par {question.get('author', 'Anonyme')} • {question.get('date', '')}",
            font=("Arial", 10),
            text_color="#888888"
        ).pack(side="left")

        # Nombre de réponses
        answers_count = len(question.get('answers', []))
        ctk.CTkLabel(
            meta_frame,
            text=f"{answers_count} réponse{'s' if answers_count != 1 else ''}",
            font=("Arial", 10),
            text_color="#00D9FF"
        ).pack(side="right")

        # Réponses (si présentes)
        answers = question.get('answers', [])
        if answers:
            for answer in answers[:2]:  # Max 2 réponses visibles
                a_frame = ctk.CTkFrame(card, fg_color="#1c2028", corner_radius=8)
                a_frame.pack(fill="x", padx=15, pady=5)

                ctk.CTkLabel(
                    a_frame,
                    text=f"R: {answer.get('content', '')}",
                    font=("Arial", 11),
                    text_color="#CCCCCC",
                    wraplength=620,
                    justify="left",
                    anchor="w"
                ).pack(anchor="w", padx=10, pady=(10, 5))

                ctk.CTkLabel(
                    a_frame,
                    text=f"— {answer.get('author', 'Anonyme')}",
                    font=("Arial", 10),
                    text_color="#00FF88"
                ).pack(anchor="e", padx=10, pady=(0, 10))

        # Bouton répondre
        ctk.CTkButton(
            card,
            text="Répondre",
            width=80,
            height=25,
            font=("Arial", 10),
            fg_color="#333333",
            hover_color="#444444",
            command=lambda q=question: self.open_answer_dialog(q)
        ).pack(anchor="e", padx=15, pady=(5, 15))

    def load_module_questions(self, module_id: str) -> list:
        """Charge les questions d'un module depuis la base"""
        try:
            import sqlite3
            from pathlib import Path

            db_path = Path(os.path.join(get_base_path(), "data", "formation_commerciale", "qa.db"))
            if not db_path.exists():
                # Créer la base si elle n'existe pas
                self._init_qa_database(db_path)
                return []

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT q.*,
                           (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
                    FROM questions q
                    WHERE q.module_id = ?
                    ORDER BY q.created_at DESC
                """, (module_id,))

                questions = []
                for row in cursor.fetchall():
                    q_dict = dict(row)
                    # Charger les réponses
                    answers_cursor = conn.execute("""
                        SELECT * FROM answers WHERE question_id = ?
                        ORDER BY created_at ASC
                    """, (q_dict['id'],))
                    q_dict['answers'] = [dict(a) for a in answers_cursor.fetchall()]
                    questions.append(q_dict)

                return questions
        except Exception as e:
            print(f"Erreur chargement Q&A: {e}")
            return []

    def _init_qa_database(self, db_path):
        """Initialise la base de données Q&A"""
        import sqlite3
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    module_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    author TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS answers (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (question_id) REFERENCES questions(id)
                )
            """)
            conn.commit()

    def open_ask_question_dialog(self):
        """Ouvre le dialogue pour poser une question"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Poser une Question")
        dialog.geometry("500x300")
        dialog.configure(fg_color="#0a0e12")
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text="Poser une Question",
            font=("Arial", 18, "bold"),
            text_color="#00BFFF"
        ).pack(pady=20)

        # Zone de texte
        question_text = ctk.CTkTextbox(
            dialog,
            height=120,
            font=("Arial", 12),
            fg_color="#1c2028"
        )
        question_text.pack(fill="x", padx=30, pady=10)

        def submit_question():
            question = question_text.get("1.0", "end-1c").strip()
            if question:
                self.save_question(question)
                dialog.destroy()
                # Rafraîchir la section Q&A
                self.master.after(100, lambda: self.master.show_module(
                    self.module, self.master.formation
                ))

        ctk.CTkButton(
            dialog,
            text="Envoyer",
            width=120,
            height=40,
            fg_color="#00D9FF",
            hover_color="#00AACC",
            command=submit_question
        ).pack(pady=20)

    def save_question(self, question_text: str):
        """Sauvegarde une question en base"""
        try:
            import sqlite3
            import uuid
            from pathlib import Path
            from datetime import datetime

            db_path = Path(os.path.join(get_base_path(), "data", "formation_commerciale", "qa.db"))
            self._init_qa_database(db_path)

            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO questions (id, module_id, question, author, date)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    self.module.get('id', ''),
                    question_text,
                    self.master.user_profile.get('username', 'Utilisateur'),
                    datetime.now().strftime("%d/%m/%Y")
                ))
                conn.commit()
        except Exception as e:
            print(f"Erreur sauvegarde question: {e}")

    def open_answer_dialog(self, question: dict):
        """Ouvre le dialogue pour répondre à une question"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Répondre")
        dialog.geometry("500x350")
        dialog.configure(fg_color="#0a0e12")
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text="Répondre à la Question",
            font=("Arial", 18, "bold"),
            text_color="#00FF88"
        ).pack(pady=15)

        # Afficher la question
        q_frame = ctk.CTkFrame(dialog, fg_color="#1c2028", corner_radius=10)
        q_frame.pack(fill="x", padx=30, pady=10)

        ctk.CTkLabel(
            q_frame,
            text=f"Q: {question.get('question', '')}",
            font=("Arial", 11),
            text_color="#CCCCCC",
            wraplength=420,
            justify="left"
        ).pack(padx=15, pady=15)

        # Zone de réponse
        answer_text = ctk.CTkTextbox(
            dialog,
            height=100,
            font=("Arial", 12),
            fg_color="#1c2028"
        )
        answer_text.pack(fill="x", padx=30, pady=10)

        def submit_answer():
            answer = answer_text.get("1.0", "end-1c").strip()
            if answer:
                self.save_answer(question.get('id'), answer)
                dialog.destroy()
                # Rafraîchir
                self.master.after(100, lambda: self.master.show_module(
                    self.module, self.master.formation
                ))

        ctk.CTkButton(
            dialog,
            text="Envoyer la Réponse",
            width=150,
            height=40,
            fg_color="#00FF88",
            hover_color="#00CC6A",
            command=submit_answer
        ).pack(pady=15)

    def save_answer(self, question_id: str, answer_text: str):
        """Sauvegarde une réponse en base"""
        try:
            import sqlite3
            import uuid
            from pathlib import Path

            db_path = Path(os.path.join(get_base_path(), "data", "formation_commerciale", "qa.db"))

            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO answers (id, question_id, content, author)
                    VALUES (?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    question_id,
                    answer_text,
                    self.master.user_profile.get('username', 'Utilisateur')
                ))
                conn.commit()
        except Exception as e:
            print(f"Erreur sauvegarde réponse: {e}")

    def show_all_questions(self, module_id: str):
        """Affiche toutes les questions dans une nouvelle fenêtre"""
        qa_win = ctk.CTkToplevel(self)
        qa_win.title("Toutes les Questions")
        qa_win.geometry("700x600")
        qa_win.configure(fg_color="#0a0e12")

        ctk.CTkLabel(
            qa_win,
            text="Toutes les Questions",
            font=("Arial", 20, "bold"),
            text_color="#00BFFF"
        ).pack(pady=20)

        scroll = ctk.CTkScrollableFrame(qa_win, fg_color="#161920")
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        questions = self.load_module_questions(module_id)
        for q in questions:
            self.create_qa_card(scroll, q)

    def create_footer(self):
        """Crée le footer avec les boutons d'action"""
        footer = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=12)
        footer.pack(fill="x", padx=20, pady=(10, 20))

        button_frame = ctk.CTkFrame(footer, fg_color="transparent")
        button_frame.pack(pady=18)

        # Bouton Marquer comme complété
        complete_btn = ctk.CTkButton(
            button_frame,
            text="Marquer comme Complété",
            font=("Helvetica", 14, "bold"),
            fg_color="#238636",
            hover_color="#2ea043",
            text_color="#FFFFFF",
            width=220,
            height=42,
            corner_radius=8,
            command=self.mark_completed
        )
        complete_btn.pack(side="left", padx=10)

        # Bouton Quiz
        quiz_btn = ctk.CTkButton(
            button_frame,
            text="Passer le Quiz",
            font=("Helvetica", 14, "bold"),
            fg_color="#1f6feb",
            hover_color="#388bfd",
            text_color="#FFFFFF",
            width=180,
            height=42,
            corner_radius=8,
            command=self.start_quiz
        )
        quiz_btn.pack(side="left", padx=10)


    def go_back(self):
        """Retour à la liste des modules"""
        self.destroy()
        self.academy.show_dashboard()

    def mark_completed(self):
        """Marque le module comme complété"""
        module_id = self.module.get('id')
        xp = self.module.get('points_xp', 0)

        self.academy.mark_module_completed(module_id, xp)
        self.go_back()

    def start_quiz(self):
        """Lance le quiz du module"""
        quiz_questions = self.module.get('quiz', [])

        if not quiz_questions:
            messagebox.showinfo("Quiz", "Aucun quiz disponible pour ce module.")
            return

        # Créer l'interface de quiz
        quiz_interface = QuizInterface(self.academy.content_area, self.module, self.academy, self)
        quiz_interface.pack(fill="both", expand=True)

        # Cacher le viewer actuel
        self.pack_forget()

class QuizInterface(ctk.CTkFrame):
    """Interface interactive de quiz"""

    def __init__(self, parent, module, academy, module_viewer):
        super().__init__(parent, fg_color="transparent")

        self.module = module
        self.academy = academy
        self.module_viewer = module_viewer
        self.quiz_questions = module.get('quiz', [])
        self.current_question = 0
        self.score = 0
        self.user_answers = []

        # Header
        header = ctk.CTkFrame(self, fg_color="#1c2028", corner_radius=15)
        header.pack(fill="x", padx=20, pady=(20, 10))

        back_btn = ctk.CTkButton(
            header,
            text="← Retour au Module",
            font=("Arial", 14),
            fg_color="#2a2d36",
            hover_color="#3a3d46",
            width=150,
            height=35,
            command=self.go_back
        )
        back_btn.pack(side="left", padx=20, pady=15)

        title = ctk.CTkLabel(
            header,
            text=f"Quiz: {module.get('titre', 'Module')}",
            font=("Arial", 22, "bold"),
            text_color="#00D9FF"
        )
        title.pack(side="left", padx=20)

        # Zone principale
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Afficher la première question
        self.display_question()

    def display_question(self):
        """Affiche une question du quiz"""
        # Nettoyer
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        if self.current_question >= len(self.quiz_questions):
            self.show_results()
            return

        question_data = self.quiz_questions[self.current_question]

        # Container de la question
        question_container = ctk.CTkFrame(self.main_frame, fg_color="#1c2028", corner_radius=15)
        question_container.pack(fill="both", expand=True)

        # Progress
        progress_text = f"Question {self.current_question + 1}/{len(self.quiz_questions)}"
        progress_label = ctk.CTkLabel(
            question_container,
            text=progress_text,
            font=("Arial", 14),
            text_color="#888888"
        )
        progress_label.pack(pady=(20, 10))

        # Barre de progression
        progress_bar = ctk.CTkProgressBar(
            question_container,
            width=400,
            progress_color="#00D9FF"
        )
        progress_bar.pack(pady=(0, 20))
        progress_bar.set((self.current_question + 1) / len(self.quiz_questions))

        # Question
        question_frame = ctk.CTkFrame(question_container, fg_color="#0f1318", corner_radius=10)
        question_frame.pack(fill="x", padx=40, pady=20)

        question_label = ctk.CTkLabel(
            question_frame,
            text=question_data.get('question', ''),
            font=("Arial", 18, "bold"),
            text_color="#FFFFFF",
            wraplength=700,
            justify="left"
        )
        question_label.pack(padx=30, pady=30)

        # Options
        self.selected_option = ctk.IntVar(value=-1)
        options = question_data.get('options', [])

        options_frame = ctk.CTkFrame(question_container, fg_color="transparent")
        options_frame.pack(fill="x", padx=60, pady=20)

        for i, option in enumerate(options):
            option_btn = ctk.CTkRadioButton(
                options_frame,
                text=option,
                variable=self.selected_option,
                value=i,
                font=("Arial", 14),
                text_color="#CCCCCC",
                fg_color="#00D9FF",
                hover_color="#00B0CC"
            )
            option_btn.pack(anchor="w", pady=8, padx=20)

        # Bouton valider
        validate_btn = ctk.CTkButton(
            question_container,
            text="Valider la Réponse",
            font=("Arial", 16, "bold"),
            fg_color="#00D9FF",
            hover_color="#00B0CC",
            width=200,
            height=50,
            command=self.validate_answer
        )
        validate_btn.pack(pady=30)

    def validate_answer(self):
        """Valide la réponse sélectionnée"""
        selected = self.selected_option.get()

        if selected == -1:
            messagebox.showwarning("Attention", "Veuillez sélectionner une réponse!")
            return

        question_data = self.quiz_questions[self.current_question]
        correct_answer = question_data.get('bonne_reponse', 0)

        # Sauvegarder la réponse
        self.user_answers.append({
            'question': question_data.get('question'),
            'selected': selected,
            'correct': correct_answer,
            'is_correct': selected == correct_answer
        })

        # Calculer le score
        if selected == correct_answer:
            self.score += 1

        # Afficher l'explication
        self.show_explanation(question_data, selected == correct_answer)

    def show_explanation(self, question_data, is_correct):
        """Affiche l'explication après la réponse"""
        # Nettoyer
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        result_container = ctk.CTkFrame(self.main_frame, fg_color="#1c2028", corner_radius=15)
        result_container.pack(fill="both", expand=True)

        # Résultat
        result_color = "#00FF88" if is_correct else "#FF6B6B"
        result_text = "✓ Correct!" if is_correct else "✗ Incorrect"

        result_label = ctk.CTkLabel(
            result_container,
            text=result_text,
            font=("Arial", 32, "bold"),
            text_color=result_color
        )
        result_label.pack(pady=(40, 20))

        # Explication
        explanation = question_data.get('explication', 'Pas d\'explication disponible.')

        explanation_frame = ctk.CTkFrame(result_container, fg_color="#0f1318", corner_radius=10)
        explanation_frame.pack(fill="x", padx=60, pady=20)

        explanation_title = ctk.CTkLabel(
            explanation_frame,
            text="Explication",
            font=("Arial", 18, "bold"),
            text_color="#00D9FF"
        )
        explanation_title.pack(pady=(20, 10), padx=30)

        explanation_text = ctk.CTkLabel(
            explanation_frame,
            text=explanation,
            font=("Arial", 14),
            text_color="#CCCCCC",
            wraplength=700,
            justify="left"
        )
        explanation_text.pack(pady=(0, 20), padx=30)

        # Bouton suivant
        next_btn = ctk.CTkButton(
            result_container,
            text="Question Suivante →",
            font=("Arial", 16, "bold"),
            fg_color="#00D9FF",
            hover_color="#00B0CC",
            width=200,
            height=50,
            command=self.next_question
        )
        next_btn.pack(pady=40)

    def next_question(self):
        """Passe à la question suivante"""
        self.current_question += 1
        self.display_question()

    def show_results(self):
        """Affiche les résultats finaux du quiz"""
        # Nettoyer
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        result_container = ctk.CTkFrame(self.main_frame, fg_color="#1c2028", corner_radius=15)
        result_container.pack(fill="both", expand=True)

        # Titre
        title_label = ctk.CTkLabel(
            result_container,
            text="🎉 Quiz Terminé!",
            font=("Arial", 32, "bold"),
            text_color="#00D9FF"
        )
        title_label.pack(pady=(40, 20))

        # Score
        percentage = int((self.score / len(self.quiz_questions)) * 100)
        score_text = f"{self.score}/{len(self.quiz_questions)}"

        score_frame = ctk.CTkFrame(result_container, fg_color="#0f1318", corner_radius=10)
        score_frame.pack(padx=60, pady=20)

        score_label = ctk.CTkLabel(
            score_frame,
            text=score_text,
            font=("Arial", 48, "bold"),
            text_color="#00FF88" if percentage >= 70 else "#FFA500" if percentage >= 50 else "#FF6B6B"
        )
        score_label.pack(pady=30)

        percentage_label = ctk.CTkLabel(
            score_frame,
            text=f"{percentage}%",
            font=("Arial", 24),
            text_color="#CCCCCC"
        )
        percentage_label.pack(pady=(0, 30))

        # Message de félicitations
        if percentage >= 80:
            message = "Excellent travail! Vous maîtrisez parfaitement ce sujet! 🌟"
            color = "#00FF88"
        elif percentage >= 60:
            message = "Bien joué! Vous avez une bonne compréhension du sujet. 👍"
            color = "#00D9FF"
        else:
            message = "Continuez vos efforts! Relisez le module pour améliorer votre score. 💪"
            color = "#FFA500"

        message_label = ctk.CTkLabel(
            result_container,
            text=message,
            font=("Arial", 16),
            text_color=color,
            wraplength=600
        )
        message_label.pack(pady=20)

        # Sauvegarder le score
        module_id = self.module.get('id')
        self.academy.user_progress['quiz_scores'][module_id] = percentage
        self.academy.save_user_progress()

        # Boutons d'action
        buttons_frame = ctk.CTkFrame(result_container, fg_color="transparent")
        buttons_frame.pack(pady=40)

        retry_btn = ctk.CTkButton(
            buttons_frame,
            text="🔄 Refaire le Quiz",
            font=("Arial", 14, "bold"),
            fg_color="#FFA500",
            hover_color="#FF8800",
            width=180,
            height=45,
            command=self.retry_quiz
        )
        retry_btn.pack(side="left", padx=10)

        back_btn = ctk.CTkButton(
            buttons_frame,
            text="← Retour au Module",
            font=("Arial", 14, "bold"),
            fg_color="#00D9FF",
            hover_color="#00B0CC",
            width=180,
            height=45,
            command=self.go_back
        )
        back_btn.pack(side="left", padx=10)

    def retry_quiz(self):
        """Recommence le quiz"""
        self.current_question = 0
        self.score = 0
        self.user_answers = []
        self.display_question()

    def go_back(self):
        """Retour au module"""
        self.destroy()
        self.module_viewer.pack(fill="both", expand=True)


def afficher_formation_commerciale(parent):
    """Point d'entrée pour lancer l'interface de formation"""
    print("[✓] Chargement HelixOne Academy v2.0...")

    try:
        # Nettoyer le parent
        for widget in parent.winfo_children():
            widget.destroy()

        # Créer l'academy
        academy = FormationAcademy(parent)

        print("[✓] HelixOne Academy chargée avec succès!")
        return academy

    except Exception as e:
        print(f"[✗] Erreur chargement academy: {e}")
        import traceback
        traceback.print_exc()

        # Afficher l'erreur
        error_frame = ctk.CTkFrame(parent, fg_color="#1c1f26")
        error_frame.pack(fill="both", expand=True, padx=20, pady=20)

        error_label = ctk.CTkLabel(
            error_frame,
            text=f"Erreur de chargement:\n{str(e)}",
            font=("Arial", 14),
            text_color="#FF6666"
        )
        error_label.pack(pady=50)

        return None


# Test autonome
if __name__ == "__main__":
    app = ctk.CTk()
    app.geometry("1400x900")
    app.title("HelixOne Academy")

    afficher_formation_commerciale(app)

    app.mainloop()
