"""
HelixOne Academy - Interface de Formation Complète
Version 2.0 - Architecture Simplifiée et Intuitive
"""

import customtkinter as ctk
from tkinter import messagebox
import json
import os
from datetime import datetime

class FormationAcademy(ctk.CTkFrame):
    """Interface principale de formation HelixOne"""

    def __init__(self, parent):
        super().__init__(parent, fg_color="#0a0e12")
        self.pack(fill="both", expand=True)

        # Chargement des données
        self.modules_data = self.load_modules()
        self.user_progress = self.load_user_progress()
        self.current_view = "dashboard"

        # Construction de l'interface
        self.build_interface()

        # Afficher le dashboard par défaut
        self.show_dashboard()

    def load_modules(self):
        """Charge tous les modules depuis le JSON"""
        try:
            json_path = "data/formation_commerciale/modules_complets.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                modules = json.load(f)

            # Organiser par parcours
            organized = {
                "debutant": [],
                "intermediaire": [],
                "expert": []
            }

            for module in modules:
                parcours = module.get("parcours", "debutant")
                if parcours in organized:
                    organized[parcours].append(module)

            print(f"[✓] {len(modules)} modules chargés")
            return organized

        except Exception as e:
            print(f"[✗] Erreur chargement modules: {e}")
            return {"debutant": [], "intermediaire": [], "expert": []}

    def load_user_progress(self):
        """Charge ou crée la progression utilisateur"""
        try:
            progress_path = "data/formation_commerciale/user_progress.json"
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                "completed_modules": [],
                "quiz_scores": {},
                "total_xp": 0,
                "level": 1,
                "current_parcours": "debutant"
            }

    def save_user_progress(self):
        """Sauvegarde la progression utilisateur"""
        try:
            progress_path = "data/formation_commerciale/user_progress.json"
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_progress, f, indent=2)
        except Exception as e:
            print(f"[✗] Erreur sauvegarde progression: {e}")

    def build_interface(self):
        """Construit l'interface principale"""
        # Header
        self.header = self.create_header()
        self.header.pack(fill="x", padx=0, pady=0)

        # Container principal
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Navigation (sidebar)
        self.sidebar = self.create_sidebar(main_container)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))

        # Zone de contenu
        self.content_area = ctk.CTkFrame(main_container, fg_color="#161920", corner_radius=15)
        self.content_area.pack(side="right", fill="both", expand=True)

    def create_header(self):
        """Crée le header de l'application"""
        header = ctk.CTkFrame(self, height=70, fg_color="#161920", corner_radius=0)
        header.pack_propagate(False)

        # Logo et titre
        title_label = ctk.CTkLabel(
            header,
            text="🎓 HelixOne Academy",
            font=("Arial", 26, "bold"),
            text_color="#00D9FF"
        )
        title_label.pack(side="left", padx=30, pady=20)

        # Stats utilisateur
        stats_frame = ctk.CTkFrame(header, fg_color="#1c2028", corner_radius=10)
        stats_frame.pack(side="right", padx=30, pady=15)

        xp_label = ctk.CTkLabel(
            stats_frame,
            text=f"⭐ {self.user_progress['total_xp']} XP",
            font=("Arial", 14, "bold"),
            text_color="#FFD700"
        )
        xp_label.pack(side="left", padx=15, pady=5)

        level_label = ctk.CTkLabel(
            stats_frame,
            text=f"🏆 Niveau {self.user_progress['level']}",
            font=("Arial", 14, "bold"),
            text_color="#00FF88"
        )
        level_label.pack(side="left", padx=15, pady=5)

        return header

    def create_sidebar(self, parent):
        """Crée la barre de navigation"""
        sidebar = ctk.CTkFrame(parent, width=250, fg_color="#161920", corner_radius=15)
        sidebar.pack_propagate(False)

        # Titre de navigation
        nav_title = ctk.CTkLabel(
            sidebar,
            text="📚 Navigation",
            font=("Arial", 16, "bold"),
            text_color="#FFFFFF"
        )
        nav_title.pack(pady=(20, 10), padx=20)

        # Séparateur
        separator = ctk.CTkFrame(sidebar, height=2, fg_color="#2a2d36")
        separator.pack(fill="x", padx=20, pady=10)

        # Boutons de navigation
        nav_items = [
            ("🏠 Dashboard", self.show_dashboard),
            ("📖 Débutant", lambda: self.show_parcours("debutant")),
            ("📊 Intermédiaire", lambda: self.show_parcours("intermediaire")),
            ("🚀 Expert", lambda: self.show_parcours("expert")),
            ("📈 Simulateur", self.show_simulateur),
            ("📚 Ma Bibliothèque", self.show_bibliotheque),
        ]

        self.nav_buttons = []
        for text, command in nav_items:
            btn = ctk.CTkButton(
                sidebar,
                text=text,
                font=("Arial", 14),
                height=45,
                fg_color="transparent",
                text_color="#CCCCCC",
                hover_color="#2a2d36",
                anchor="w",
                command=command
            )
            btn.pack(fill="x", padx=15, pady=3)
            self.nav_buttons.append(btn)

        return sidebar

    def clear_content(self):
        """Nettoie la zone de contenu"""
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def highlight_nav_button(self, index):
        """Met en surbrillance le bouton actif"""
        for i, btn in enumerate(self.nav_buttons):
            if i == index:
                btn.configure(fg_color="#00D9FF", text_color="#000000")
            else:
                btn.configure(fg_color="transparent", text_color="#CCCCCC")

    def show_dashboard(self):
        """Affiche le dashboard principal"""
        self.clear_content()
        self.highlight_nav_button(0)
        self.current_view = "dashboard"

        # Scroll frame
        scroll = ctk.CTkScrollableFrame(
            self.content_area,
            fg_color="transparent"
        )
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        title = ctk.CTkLabel(
            scroll,
            text="📊 Votre Progression",
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

        stats = [
            ("Modules Complétés", f"{completed}/{total_modules}", "#00FF88"),
            ("Progression Globale", f"{progress_pct}%", "#00D9FF"),
            ("XP Total", f"{self.user_progress['total_xp']}", "#FFD700"),
            ("Niveau Actuel", f"{self.user_progress['level']}", "#FF6B6B")
        ]

        for i, (label, value, color) in enumerate(stats):
            card = self.create_stat_card(stats_container, label, value, color)
            card.grid(row=0, column=i, padx=10, sticky="ew")
            stats_container.grid_columnconfigure(i, weight=1)

        # Parcours disponibles
        parcours_title = ctk.CTkLabel(
            scroll,
            text="🎯 Parcours de Formation",
            font=("Arial", 22, "bold"),
            text_color="#FFFFFF"
        )
        parcours_title.pack(pady=(30, 20))

        # Cards des parcours
        for parcours_id, parcours_name, color, icon in [
            ("debutant", "Trader Débutant", "#00D9FF", "📖"),
            ("intermediaire", "Trader Confirmé", "#FFA500", "📊"),
            ("expert", "Trader Expert", "#FF4444", "🚀")
        ]:
            self.create_parcours_card(scroll, parcours_id, parcours_name, color, icon)

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

    def create_parcours_card(self, parent, parcours_id, parcours_name, color, icon):
        """Crée une carte de parcours"""
        modules = self.modules_data.get(parcours_id, [])
        completed = len([m for m in modules if m.get('id') in self.user_progress['completed_modules']])
        total = len(modules)
        progress = int((completed / total * 100)) if total > 0 else 0

        card = ctk.CTkFrame(parent, fg_color="#1c2028", corner_radius=15)
        card.pack(fill="x", pady=10)

        # Header
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))

        title = ctk.CTkLabel(
            header_frame,
            text=f"{icon} {parcours_name}",
            font=("Arial", 20, "bold"),
            text_color=color
        )
        title.pack(side="left")

        status = ctk.CTkLabel(
            header_frame,
            text=f"{completed}/{total} modules",
            font=("Arial", 14),
            text_color="#888888"
        )
        status.pack(side="right")

        # Barre de progression
        progress_frame = ctk.CTkFrame(card, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=(0, 10))

        progress_bar = ctk.CTkProgressBar(
            progress_frame,
            height=10,
            progress_color=color
        )
        progress_bar.pack(fill="x")
        progress_bar.set(progress / 100)

        progress_label = ctk.CTkLabel(
            progress_frame,
            text=f"{progress}% complété",
            font=("Arial", 12),
            text_color="#CCCCCC"
        )
        progress_label.pack(pady=(5, 0))

        # Bouton d'action
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
        """Affiche les modules d'un parcours"""
        self.clear_content()

        # Déterminer quel bouton highlighter
        nav_index = {"debutant": 1, "intermediaire": 2, "expert": 3}.get(parcours_id, 0)
        self.highlight_nav_button(nav_index)

        self.current_view = f"parcours_{parcours_id}"

        modules = self.modules_data.get(parcours_id, [])

        if not modules:
            # Pas de modules
            empty_label = ctk.CTkLabel(
                self.content_area,
                text=f"Aucun module disponible pour ce parcours",
                font=("Arial", 16),
                text_color="#888888"
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
        parcours_names = {
            "debutant": "📖 Formation Débutant",
            "intermediaire": "📊 Formation Intermédiaire",
            "expert": "🚀 Formation Expert"
        }

        title = ctk.CTkLabel(
            scroll,
            text=parcours_names.get(parcours_id, "Formation"),
            font=("Arial", 28, "bold"),
            text_color="#00D9FF"
        )
        title.pack(pady=(0, 30))

        # Afficher chaque module
        for i, module in enumerate(modules, 1):
            self.create_module_card(scroll, module, i)

    def create_module_card(self, parent, module, index):
        """Crée une carte de module"""
        module_id = module.get('id', '')
        is_completed = module_id in self.user_progress['completed_modules']

        card = ctk.CTkFrame(parent, fg_color="#1c2028", corner_radius=15)
        card.pack(fill="x", pady=10)

        # Container principal
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=20, pady=20)

        # Numéro et statut
        left_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        left_frame.pack(side="left", anchor="n")

        number_label = ctk.CTkLabel(
            left_frame,
            text=f"{index}",
            font=("Arial", 32, "bold"),
            text_color="#00D9FF" if not is_completed else "#00FF88",
            width=50
        )
        number_label.pack()

        if is_completed:
            check_label = ctk.CTkLabel(
                left_frame,
                text="✓",
                font=("Arial", 24, "bold"),
                text_color="#00FF88"
            )
            check_label.pack()

        # Détails du module
        details_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        details_frame.pack(side="left", fill="both", expand=True, padx=(20, 0))

        # Titre
        title_label = ctk.CTkLabel(
            details_frame,
            text=module.get('titre', 'Module sans titre'),
            font=("Arial", 18, "bold"),
            text_color="#FFFFFF",
            anchor="w"
        )
        title_label.pack(fill="x")

        # Description
        desc_label = ctk.CTkLabel(
            details_frame,
            text=module.get('description', ''),
            font=("Arial", 13),
            text_color="#AAAAAA",
            anchor="w",
            wraplength=600,
            justify="left"
        )
        desc_label.pack(fill="x", pady=(5, 10))

        # Infos (durée, difficulté, XP)
        info_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        info_frame.pack(fill="x")

        infos = [
            f"⏱️ {module.get('durée', 'N/A')}",
            f"📊 {module.get('difficulté', 'N/A')}",
            f"⭐ {module.get('points_xp', 0)} XP"
        ]

        for info in infos:
            info_label = ctk.CTkLabel(
                info_frame,
                text=info,
                font=("Arial", 12),
                text_color="#888888"
            )
            info_label.pack(side="left", padx=(0, 20))

        # Bouton d'action
        btn_text = "Revoir" if is_completed else "Commencer"
        btn_color = "#00FF88" if is_completed else "#00D9FF"

        action_btn = ctk.CTkButton(
            content_frame,
            text=btn_text,
            font=("Arial", 14, "bold"),
            fg_color=btn_color,
            hover_color=btn_color,
            width=120,
            height=40,
            command=lambda: self.open_module(module)
        )
        action_btn.pack(side="right", anchor="n")

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

            # Mettre à jour le header
            self.build_interface()

            messagebox.showinfo(
                "Module Complété!",
                f"Félicitations! Vous avez gagné {xp_earned} XP!\n\nTotal XP: {self.user_progress['total_xp']}"
            )

    def show_simulateur(self):
        """Affiche le simulateur de trading"""
        self.clear_content()
        self.highlight_nav_button(4)

        # Créer le simulateur
        simulateur = SimulateurTrading(self.content_area, self)
        simulateur.pack(fill="both", expand=True)

    def show_bibliotheque(self):
        """Affiche la bibliothèque de ressources"""
        self.clear_content()
        self.highlight_nav_button(5)

        scroll = ctk.CTkScrollableFrame(
            self.content_area,
            fg_color="transparent"
        )
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        title = ctk.CTkLabel(
            scroll,
            text="📚 Bibliothèque de Ressources",
            font=("Arial", 28, "bold"),
            text_color="#00D9FF"
        )
        title.pack(pady=(0, 30))

        # Ressources par catégorie
        categories = {
            "📄 Articles": [
                "Guide complet de l'Analyse Technique",
                "Les 10 erreurs du débutant en trading",
                "Stratégies de gestion du risque",
                "Psychologie du trader gagnant"
            ],
            "🎥 Vidéos": [
                "Webinaire: Market Making expliqué",
                "Tutorial: Configuration TradingView",
                "Analyse en direct du CAC40",
                "Les indicateurs techniques essentiels"
            ],
            "🛠️ Outils": [
                "Calculateur de taille de position",
                "Risk/Reward Calculator",
                "Backtesting Tool",
                "Journal de trading"
            ]
        }

        for category, items in categories.items():
            # Titre de catégorie
            cat_label = ctk.CTkLabel(
                scroll,
                text=category,
                font=("Arial", 20, "bold"),
                text_color="#FFFFFF"
            )
            cat_label.pack(anchor="w", pady=(20, 10))

            # Items de la catégorie
            for item in items:
                item_frame = ctk.CTkFrame(scroll, fg_color="#1c2028", corner_radius=10)
                item_frame.pack(fill="x", pady=5)

                item_label = ctk.CTkLabel(
                    item_frame,
                    text=f"• {item}",
                    font=("Arial", 14),
                    text_color="#CCCCCC",
                    anchor="w"
                )
                item_label.pack(side="left", padx=20, pady=15)

                download_btn = ctk.CTkButton(
                    item_frame,
                    text="Télécharger",
                    font=("Arial", 12),
                    fg_color="#00D9FF",
                    width=100,
                    height=30
                )
                download_btn.pack(side="right", padx=20)


class ModuleViewer(ctk.CTkFrame):
    """Visualiseur de module complet"""

    def __init__(self, parent, module, academy):
        super().__init__(parent, fg_color="transparent")

        self.module = module
        self.academy = academy

        # Header avec retour
        header = ctk.CTkFrame(self, fg_color="#1c2028", corner_radius=15)
        header.pack(fill="x", padx=20, pady=(20, 10))

        back_btn = ctk.CTkButton(
            header,
            text="← Retour",
            font=("Arial", 14),
            fg_color="#2a2d36",
            hover_color="#3a3d46",
            width=100,
            height=35,
            command=self.go_back
        )
        back_btn.pack(side="left", padx=20, pady=15)

        title = ctk.CTkLabel(
            header,
            text=module.get('titre', 'Module'),
            font=("Arial", 22, "bold"),
            text_color="#00D9FF"
        )
        title.pack(side="left", padx=20)

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
        """Affiche le contenu complet du module"""
        contenu = self.module.get('contenu', {})

        # Introduction
        intro = contenu.get('introduction', '')
        if intro:
            intro_frame = ctk.CTkFrame(self.scroll, fg_color="#1c2028", corner_radius=15)
            intro_frame.pack(fill="x", pady=10)

            intro_title = ctk.CTkLabel(
                intro_frame,
                text="📖 Introduction",
                font=("Arial", 18, "bold"),
                text_color="#00D9FF"
            )
            intro_title.pack(anchor="w", padx=20, pady=(20, 10))

            intro_text = ctk.CTkTextbox(
                intro_frame,
                font=("Arial", 13),
                wrap="word",
                height=120,
                fg_color="#0f1318"
            )
            intro_text.pack(fill="x", padx=20, pady=(0, 20))
            intro_text.insert("1.0", intro)
            intro_text.configure(state="disabled")

        # Sections
        sections = contenu.get('sections', [])
        for i, section in enumerate(sections, 1):
            section_frame = ctk.CTkFrame(self.scroll, fg_color="#1c2028", corner_radius=15)
            section_frame.pack(fill="x", pady=10)

            # Titre de section
            section_title = ctk.CTkLabel(
                section_frame,
                text=f"{i}. {section.get('titre', 'Section')}",
                font=("Arial", 18, "bold"),
                text_color="#FFA500"
            )
            section_title.pack(anchor="w", padx=20, pady=(20, 10))

            # Contenu de section
            section_content = section.get('contenu', '')
            if section_content:
                content_text = ctk.CTkTextbox(
                    section_frame,
                    font=("Arial", 13),
                    wrap="word",
                    height=300,
                    fg_color="#0f1318"
                )
                content_text.pack(fill="x", padx=20, pady=(0, 15))
                content_text.insert("1.0", section_content)
                content_text.configure(state="disabled")

            # Points clés
            points_cles = section.get('points_cles', [])
            if points_cles:
                points_frame = ctk.CTkFrame(section_frame, fg_color="#0f1318", corner_radius=10)
                points_frame.pack(fill="x", padx=20, pady=(0, 20))

                points_title = ctk.CTkLabel(
                    points_frame,
                    text="💡 Points Clés à Retenir",
                    font=("Arial", 14, "bold"),
                    text_color="#00FF88"
                )
                points_title.pack(anchor="w", padx=15, pady=(15, 10))

                for point in points_cles:
                    point_label = ctk.CTkLabel(
                        points_frame,
                        text=f"✓ {point}",
                        font=("Arial", 12),
                        text_color="#CCCCCC",
                        anchor="w",
                        wraplength=700,
                        justify="left"
                    )
                    point_label.pack(anchor="w", padx=30, pady=3)

                ctk.CTkLabel(points_frame, text="").pack(pady=5)

        # Résumé
        resume = contenu.get('resume', '')
        if resume:
            resume_frame = ctk.CTkFrame(self.scroll, fg_color="#1c2028", corner_radius=15)
            resume_frame.pack(fill="x", pady=10)

            resume_title = ctk.CTkLabel(
                resume_frame,
                text="📝 Résumé",
                font=("Arial", 18, "bold"),
                text_color="#00D9FF"
            )
            resume_title.pack(anchor="w", padx=20, pady=(20, 10))

            resume_text = ctk.CTkTextbox(
                resume_frame,
                font=("Arial", 13),
                wrap="word",
                height=100,
                fg_color="#0f1318"
            )
            resume_text.pack(fill="x", padx=20, pady=(0, 20))
            resume_text.insert("1.0", resume)
            resume_text.configure(state="disabled")

        # Ressources complémentaires
        ressources = contenu.get('ressources_complementaires', [])
        if ressources:
            ressources_frame = ctk.CTkFrame(self.scroll, fg_color="#1c2028", corner_radius=15)
            ressources_frame.pack(fill="x", pady=10)

            ressources_title = ctk.CTkLabel(
                ressources_frame,
                text="📚 Ressources Complémentaires",
                font=("Arial", 16, "bold"),
                text_color="#FFA500"
            )
            ressources_title.pack(anchor="w", padx=20, pady=(20, 10))

            for ressource in ressources:
                ressource_label = ctk.CTkLabel(
                    ressources_frame,
                    text=f"• {ressource}",
                    font=("Arial", 12),
                    text_color="#CCCCCC",
                    anchor="w"
                )
                ressource_label.pack(anchor="w", padx=30, pady=3)

            ctk.CTkLabel(ressources_frame, text="").pack(pady=10)

    def create_footer(self):
        """Crée le footer avec les boutons d'action"""
        footer = ctk.CTkFrame(self, fg_color="#1c2028", corner_radius=15)
        footer.pack(fill="x", padx=20, pady=(10, 20))

        button_frame = ctk.CTkFrame(footer, fg_color="transparent")
        button_frame.pack(pady=20)

        # Bouton Marquer comme complété
        complete_btn = ctk.CTkButton(
            button_frame,
            text="✓ Marquer comme Complété",
            font=("Arial", 14, "bold"),
            fg_color="#00FF88",
            hover_color="#00CC6A",
            width=220,
            height=45,
            command=self.mark_completed
        )
        complete_btn.pack(side="left", padx=10)

        # Bouton Quiz
        quiz_btn = ctk.CTkButton(
            button_frame,
            text="📝 Passer le Quiz",
            font=("Arial", 14, "bold"),
            fg_color="#00D9FF",
            hover_color="#00B0CC",
            width=180,
            height=45,
            command=self.start_quiz
        )
        quiz_btn.pack(side="left", padx=10)

        # Bouton Exercices
        if self.module.get('exercices'):
            exercices_btn = ctk.CTkButton(
                button_frame,
                text="✏️ Exercices",
                font=("Arial", 14, "bold"),
                fg_color="#FFA500",
                hover_color="#FF8800",
                width=150,
                height=45,
                command=self.show_exercices
            )
            exercices_btn.pack(side="left", padx=10)

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

        # TODO: Implémenter l'interface de quiz
        messagebox.showinfo(
            "Quiz",
            f"Quiz disponible avec {len(quiz_questions)} questions!\n\nCette fonctionnalité sera bientôt disponible."
        )

    def show_exercices(self):
        """Affiche les exercices du module"""
        exercices = self.module.get('exercices', [])

        if not exercices:
            messagebox.showinfo("Exercices", "Aucun exercice disponible pour ce module.")
            return

        # TODO: Implémenter l'interface d'exercices
        messagebox.showinfo(
            "Exercices",
            f"{len(exercices)} exercices disponibles!\n\nCette fonctionnalité sera bientôt disponible."
        )


class SimulateurTrading(ctk.CTkFrame):
    """Simulateur de trading simplifié et intuitif"""

    def __init__(self, parent, academy):
        super().__init__(parent, fg_color="transparent")

        self.academy = academy
        self.portfolio = {
            "cash": 10000.0,
            "positions": {}
        }

        # Titre
        title_frame = ctk.CTkFrame(self, fg_color="#1c2028", corner_radius=15)
        title_frame.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            title_frame,
            text="📈 Simulateur de Trading",
            font=("Arial", 28, "bold"),
            text_color="#00D9FF"
        )
        title.pack(pady=20)

        # Container principal
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20)

        # Colonne gauche: Portfolio
        left_col = ctk.CTkFrame(main_container, fg_color="#1c2028", corner_radius=15)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.create_portfolio_section(left_col)

        # Colonne droite: Trading
        right_col = ctk.CTkFrame(main_container, fg_color="#1c2028", corner_radius=15)
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.create_trading_section(right_col)

    def create_portfolio_section(self, parent):
        """Section portfolio"""
        title = ctk.CTkLabel(
            parent,
            text="💼 Votre Portfolio",
            font=("Arial", 20, "bold"),
            text_color="#00FF88"
        )
        title.pack(pady=(20, 15), padx=20)

        # Cash disponible
        cash_frame = ctk.CTkFrame(parent, fg_color="#0f1318", corner_radius=10)
        cash_frame.pack(fill="x", padx=20, pady=10)

        cash_label = ctk.CTkLabel(
            cash_frame,
            text="💵 Cash Disponible",
            font=("Arial", 14),
            text_color="#AAAAAA"
        )
        cash_label.pack(pady=(15, 5))

        self.cash_value = ctk.CTkLabel(
            cash_frame,
            text=f"{self.portfolio['cash']:,.2f} €",
            font=("Arial", 24, "bold"),
            text_color="#00FF88"
        )
        self.cash_value.pack(pady=(0, 15))

        # Valeur totale
        total_frame = ctk.CTkFrame(parent, fg_color="#0f1318", corner_radius=10)
        total_frame.pack(fill="x", padx=20, pady=10)

        total_label = ctk.CTkLabel(
            total_frame,
            text="📊 Valeur Totale",
            font=("Arial", 14),
            text_color="#AAAAAA"
        )
        total_label.pack(pady=(15, 5))

        self.total_value = ctk.CTkLabel(
            total_frame,
            text=f"{self.portfolio['cash']:,.2f} €",
            font=("Arial", 24, "bold"),
            text_color="#00D9FF"
        )
        self.total_value.pack(pady=(0, 15))

        # Positions
        positions_title = ctk.CTkLabel(
            parent,
            text="📋 Positions Ouvertes",
            font=("Arial", 16, "bold"),
            text_color="#FFFFFF"
        )
        positions_title.pack(pady=(20, 10), padx=20)

        self.positions_list = ctk.CTkScrollableFrame(
            parent,
            fg_color="#0f1318",
            corner_radius=10,
            height=200
        )
        self.positions_list.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Message si pas de positions
        no_positions = ctk.CTkLabel(
            self.positions_list,
            text="Aucune position ouverte",
            font=("Arial", 13),
            text_color="#888888"
        )
        no_positions.pack(pady=30)

    def create_trading_section(self, parent):
        """Section trading"""
        title = ctk.CTkLabel(
            parent,
            text="🎯 Passer un Ordre",
            font=("Arial", 20, "bold"),
            text_color="#00D9FF"
        )
        title.pack(pady=(20, 15), padx=20)

        # Formulaire d'ordre
        form_frame = ctk.CTkFrame(parent, fg_color="#0f1318", corner_radius=10)
        form_frame.pack(fill="x", padx=20, pady=10)

        # Symbole
        ctk.CTkLabel(
            form_frame,
            text="Symbole (ex: AAPL, TSLA, MSFT)",
            font=("Arial", 13, "bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w", padx=20, pady=(20, 5))

        self.symbol_entry = ctk.CTkEntry(
            form_frame,
            font=("Arial", 14),
            height=40,
            placeholder_text="AAPL"
        )
        self.symbol_entry.pack(fill="x", padx=20, pady=(0, 15))

        # Quantité
        ctk.CTkLabel(
            form_frame,
            text="Quantité",
            font=("Arial", 13, "bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w", padx=20, pady=(10, 5))

        self.quantity_entry = ctk.CTkEntry(
            form_frame,
            font=("Arial", 14),
            height=40,
            placeholder_text="10"
        )
        self.quantity_entry.pack(fill="x", padx=20, pady=(0, 15))

        # Prix (simulé)
        ctk.CTkLabel(
            form_frame,
            text="Prix par action (€)",
            font=("Arial", 13, "bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w", padx=20, pady=(10, 5))

        self.price_entry = ctk.CTkEntry(
            form_frame,
            font=("Arial", 14),
            height=40,
            placeholder_text="150.00"
        )
        self.price_entry.pack(fill="x", padx=20, pady=(0, 20))

        # Boutons Acheter / Vendre
        buttons_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 20))

        buy_btn = ctk.CTkButton(
            buttons_frame,
            text="🟢 ACHETER",
            font=("Arial", 16, "bold"),
            fg_color="#00FF88",
            hover_color="#00CC6A",
            height=50,
            command=lambda: self.place_order("BUY")
        )
        buy_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        sell_btn = ctk.CTkButton(
            buttons_frame,
            text="🔴 VENDRE",
            font=("Arial", 16, "bold"),
            fg_color="#FF6B6B",
            hover_color="#FF5252",
            height=50,
            command=lambda: self.place_order("SELL")
        )
        sell_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Historique
        history_title = ctk.CTkLabel(
            parent,
            text="📜 Historique des Ordres",
            font=("Arial", 16, "bold"),
            text_color="#FFFFFF"
        )
        history_title.pack(pady=(20, 10), padx=20)

        self.history_list = ctk.CTkScrollableFrame(
            parent,
            fg_color="#0f1318",
            corner_radius=10,
            height=200
        )
        self.history_list.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Message si pas d'historique
        no_history = ctk.CTkLabel(
            self.history_list,
            text="Aucun ordre passé",
            font=("Arial", 13),
            text_color="#888888"
        )
        no_history.pack(pady=30)

    def place_order(self, order_type):
        """Place un ordre d'achat ou de vente"""
        symbol = self.symbol_entry.get().strip().upper()
        quantity_str = self.quantity_entry.get().strip()
        price_str = self.price_entry.get().strip()

        # Validation
        if not symbol:
            messagebox.showerror("Erreur", "Veuillez entrer un symbole")
            return

        try:
            quantity = int(quantity_str)
            if quantity <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Erreur", "Quantité invalide")
            return

        try:
            price = float(price_str)
            if price <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Erreur", "Prix invalide")
            return

        total_cost = quantity * price

        if order_type == "BUY":
            if total_cost > self.portfolio['cash']:
                messagebox.showerror(
                    "Fonds Insuffisants",
                    f"Coût total: {total_cost:,.2f} €\nCash disponible: {self.portfolio['cash']:,.2f} €"
                )
                return

            # Exécuter l'achat
            self.portfolio['cash'] -= total_cost

            if symbol in self.portfolio['positions']:
                self.portfolio['positions'][symbol] += quantity
            else:
                self.portfolio['positions'][symbol] = quantity

            messagebox.showinfo(
                "Ordre Exécuté",
                f"✓ Achat de {quantity} actions {symbol}\n"
                f"Prix: {price:.2f} €\n"
                f"Total: {total_cost:,.2f} €"
            )

        else:  # SELL
            if symbol not in self.portfolio['positions'] or self.portfolio['positions'][symbol] < quantity:
                messagebox.showerror(
                    "Position Insuffisante",
                    f"Vous ne possédez pas assez d'actions {symbol}"
                )
                return

            # Exécuter la vente
            self.portfolio['cash'] += total_cost
            self.portfolio['positions'][symbol] -= quantity

            if self.portfolio['positions'][symbol] == 0:
                del self.portfolio['positions'][symbol]

            messagebox.showinfo(
                "Ordre Exécuté",
                f"✓ Vente de {quantity} actions {symbol}\n"
                f"Prix: {price:.2f} €\n"
                f"Total: {total_cost:,.2f} €"
            )

        # Mettre à jour l'affichage
        self.update_portfolio_display()
        self.add_to_history(order_type, symbol, quantity, price)

        # Nettoyer le formulaire
        self.symbol_entry.delete(0, 'end')
        self.quantity_entry.delete(0, 'end')
        self.price_entry.delete(0, 'end')

    def update_portfolio_display(self):
        """Met à jour l'affichage du portfolio"""
        self.cash_value.configure(text=f"{self.portfolio['cash']:,.2f} €")

        # Calculer valeur totale (cash + positions)
        total = self.portfolio['cash']
        # TODO: ajouter la valeur des positions

        self.total_value.configure(text=f"{total:,.2f} €")

        # Mettre à jour la liste des positions
        for widget in self.positions_list.winfo_children():
            widget.destroy()

        if not self.portfolio['positions']:
            no_positions = ctk.CTkLabel(
                self.positions_list,
                text="Aucune position ouverte",
                font=("Arial", 13),
                text_color="#888888"
            )
            no_positions.pack(pady=30)
        else:
            for symbol, qty in self.portfolio['positions'].items():
                pos_frame = ctk.CTkFrame(self.positions_list, fg_color="#1c2028", corner_radius=8)
                pos_frame.pack(fill="x", pady=5, padx=10)

                pos_label = ctk.CTkLabel(
                    pos_frame,
                    text=f"{symbol}: {qty} actions",
                    font=("Arial", 13, "bold"),
                    text_color="#00D9FF"
                )
                pos_label.pack(pady=10, padx=15)

    def add_to_history(self, order_type, symbol, quantity, price):
        """Ajoute un ordre à l'historique"""
        # Nettoyer le message "Aucun ordre passé" si présent
        for widget in self.history_list.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and "Aucun ordre" in widget.cget("text"):
                widget.destroy()

        # Ajouter le nouvel ordre
        history_frame = ctk.CTkFrame(self.history_list, fg_color="#1c2028", corner_radius=8)
        history_frame.pack(fill="x", pady=5, padx=10)

        color = "#00FF88" if order_type == "BUY" else "#FF6B6B"
        type_text = "ACHAT" if order_type == "BUY" else "VENTE"

        timestamp = datetime.now().strftime("%H:%M:%S")

        history_label = ctk.CTkLabel(
            history_frame,
            text=f"[{timestamp}] {type_text} {quantity} {symbol} @ {price:.2f}€",
            font=("Arial", 12),
            text_color=color
        )
        history_label.pack(pady=8, padx=15)


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
