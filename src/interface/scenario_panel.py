"""
Panel de Simulation de Scénarios
Interface pour le Scenario Engine (inspiré de BlackRock Aladdin)
"""

import customtkinter as ctk
from typing import Dict, Optional, List
import logging
import threading
import json

# Configuration des couleurs
COLORS = {
    'bg_primary': ("#F5F5F5", "#1a1a1a"),
    'bg_secondary': ("#FFFFFF", "#2b2b2b"),
    'bg_tertiary': ("#E8E8E8", "#333333"),
    'accent_blue': "#3b8ed0",
    'accent_green': "#2ecc71",
    'accent_red': "#e74c3c",
    'accent_orange': "#f39c12",
    'text_primary': ("#1a1a1a", "#FFFFFF"),
    'text_secondary': ("#666666", "#999999"),
}

logger = logging.getLogger(__name__)


class ScenarioPanel(ctk.CTkFrame):
    """
    Panel principal pour la simulation de scénarios
    """

    def __init__(self, master, helixone_client=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self.helixone_client = helixone_client

        # État
        self.predefined_scenarios = None
        self.current_result = None

        # Configuration des colonnes/lignes
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Créer l'interface
        self._create_header()
        self._create_main_layout()

        # Charger les scénarios prédéfinis
        self._load_predefined_scenarios()

    def _create_header(self):
        """Créer le header avec titre et boutons"""
        header_frame = ctk.CTkFrame(self, fg_color=COLORS['bg_secondary'])
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        # Titre
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎯 Simulation de Scénarios",
            font=("Segoe UI", 28, "bold"),
            text_color=COLORS['accent_blue']
        )
        title_label.pack(side="left", padx=20, pady=15)

        # Sous-titre
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Testez la résilience de votre portfolio face à différents scénarios de marché",
            font=("Segoe UI", 12),
            text_color=COLORS['text_secondary']
        )
        subtitle_label.pack(side="left", padx=(0, 20))

        # Bouton rafraîchir
        refresh_btn = ctk.CTkButton(
            header_frame,
            text="🔄 Rafraîchir",
            width=120,
            height=35,
            fg_color=COLORS['bg_tertiary'],
            hover_color=COLORS['accent_blue'],
            command=self._load_predefined_scenarios
        )
        refresh_btn.pack(side="right", padx=20)

    def _create_main_layout(self):
        """Créer le layout principal"""
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=2)
        main_container.grid_rowconfigure(0, weight=1)

        # Panel gauche: Sélection de scénario
        self.selection_panel = self._create_selection_panel(main_container)
        self.selection_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Panel droit: Résultats
        self.results_panel = self._create_results_panel(main_container)
        self.results_panel.grid(row=0, column=1, sticky="nsew")

    def _create_selection_panel(self, parent):
        """Panel de sélection de scénario"""
        panel = ctk.CTkFrame(parent, fg_color=COLORS['bg_secondary'])

        # Titre
        title_label = ctk.CTkLabel(
            panel,
            text="📋 Choisir un Scénario",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        )
        title_label.pack(fill="x", padx=20, pady=(20, 15))

        # Scrollable frame pour les scénarios
        scroll_frame = ctk.CTkScrollableFrame(
            panel,
            fg_color="transparent"
        )
        scroll_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.scenarios_container = scroll_frame

        return panel

    def _create_results_panel(self, parent):
        """Panel d'affichage des résultats"""
        panel = ctk.CTkFrame(parent, fg_color=COLORS['bg_secondary'])

        # Titre
        title_label = ctk.CTkLabel(
            panel,
            text="📊 Résultats de Simulation",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        )
        title_label.pack(fill="x", padx=20, pady=(20, 15))

        # Scrollable frame pour les résultats
        scroll_frame = ctk.CTkScrollableFrame(
            panel,
            fg_color="transparent"
        )
        scroll_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Message par défaut
        default_msg = ctk.CTkLabel(
            scroll_frame,
            text="👈 Sélectionnez un scénario pour commencer",
            font=("Segoe UI", 14),
            text_color=COLORS['text_secondary']
        )
        default_msg.pack(pady=100)

        self.results_container = scroll_frame

        return panel

    def _load_predefined_scenarios(self):
        """Charger les scénarios prédéfinis depuis l'API"""
        logger.info("Chargement des scénarios prédéfinis...")

        def load_in_background():
            try:
                if not self.helixone_client:
                    raise Exception("Client API non configuré")

                # Appeler l'API
                response = self.helixone_client._make_request(
                    "GET",
                    "/api/scenarios/predefined",
                    require_auth=True
                )

                self.predefined_scenarios = response

                # Mettre à jour l'UI dans le thread principal
                self.after(0, self._display_scenarios)

            except Exception as e:
                logger.error(f"Erreur chargement scénarios: {e}")
                self.after(0, lambda: self._show_error(str(e)))

        thread = threading.Thread(target=load_in_background, daemon=True)
        thread.start()

    def _display_scenarios(self):
        """Afficher les scénarios dans le panel de sélection"""
        # Clear existing
        for widget in self.scenarios_container.winfo_children():
            widget.destroy()

        if not self.predefined_scenarios:
            error_label = ctk.CTkLabel(
                self.scenarios_container,
                text="❌ Impossible de charger les scénarios",
                text_color=COLORS['accent_red']
            )
            error_label.pack(pady=20)
            return

        # Section 1: Stress Tests
        stress_tests = self.predefined_scenarios.get('stress_tests', [])
        if stress_tests:
            self._create_scenario_section(
                "💥 Stress Tests Standards",
                stress_tests,
                "stress_test"
            )

        # Section 2: Événements Historiques
        historical = self.predefined_scenarios.get('historical_events', [])
        if historical:
            self._create_scenario_section(
                "📜 Événements Historiques",
                historical,
                "historical"
            )

    def _create_scenario_section(self, title: str, scenarios: List, scenario_type: str):
        """Créer une section de scénarios"""
        # Titre de section
        section_title = ctk.CTkLabel(
            self.scenarios_container,
            text=title,
            font=("Segoe UI", 14, "bold"),
            anchor="w"
        )
        section_title.pack(fill="x", padx=10, pady=(15, 10))

        # Cartes de scénarios
        for scenario in scenarios:
            self._create_scenario_card(scenario, scenario_type)

    def _create_scenario_card(self, scenario: Dict, scenario_type: str):
        """Créer une carte pour un scénario"""
        card = ctk.CTkFrame(
            self.scenarios_container,
            fg_color=COLORS['bg_tertiary'],
            corner_radius=10
        )
        card.pack(fill="x", padx=10, pady=5)

        # Nom
        name_label = ctk.CTkLabel(
            card,
            text=scenario['name'],
            font=("Segoe UI", 13, "bold"),
            anchor="w"
        )
        name_label.pack(fill="x", padx=15, pady=(15, 5))

        # Description
        desc = scenario.get('description', '')
        if desc:
            desc_label = ctk.CTkLabel(
                card,
                text=desc,
                font=("Segoe UI", 10),
                text_color=COLORS['text_secondary'],
                anchor="w",
                wraplength=280
            )
            desc_label.pack(fill="x", padx=15, pady=(0, 10))

        # Bouton simuler
        simulate_btn = ctk.CTkButton(
            card,
            text="▶ Simuler",
            width=100,
            height=30,
            fg_color=COLORS['accent_blue'],
            hover_color=COLORS['accent_green'],
            command=lambda: self._run_scenario(scenario, scenario_type)
        )
        simulate_btn.pack(pady=(0, 15), padx=15)

    def _run_scenario(self, scenario: Dict, scenario_type: str):
        """Lancer une simulation de scénario"""
        logger.info(f"Lancement simulation: {scenario['name']}")

        # Afficher loading
        self._show_loading()

        def run_in_background():
            try:
                if not self.helixone_client:
                    raise Exception("Client API non configuré")

                # Récupérer le portfolio de test
                # TODO: Permettre à l'utilisateur de sélectionner son portfolio
                test_portfolio = {
                    "AAPL": 100,
                    "MSFT": 50,
                    "GOOGL": 30,
                    "TSLA": 20
                }

                # Préparer la requête selon le type
                if scenario_type == "stress_test":
                    params = scenario.get('parameters', {})
                    payload = {
                        "portfolio": test_portfolio,
                        "stress_test_type": scenario['type'],
                        **params
                    }
                    endpoint = "/api/scenarios/stress-test"

                elif scenario_type == "historical":
                    payload = {
                        "portfolio": test_portfolio,
                        "event_name": scenario['name']
                    }
                    endpoint = "/api/scenarios/historical"

                else:
                    raise Exception(f"Type de scénario non supporté: {scenario_type}")

                # Appeler l'API
                result = self.helixone_client._make_request(
                    "POST",
                    endpoint,
                    data=payload,
                    require_auth=True
                )

                self.current_result = result

                # Afficher les résultats
                self.after(0, self._display_results)

            except Exception as e:
                logger.error(f"Erreur simulation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.after(0, lambda: self._show_error(str(e)))

        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()

    def _show_loading(self):
        """Afficher un indicateur de chargement"""
        # Clear
        for widget in self.results_container.winfo_children():
            widget.destroy()

        loading_label = ctk.CTkLabel(
            self.results_container,
            text="⏳ Simulation en cours...\n\nCollecte des données de marché...\nCalcul des impacts...\nGénération des recommandations...",
            font=("Segoe UI", 14),
            text_color=COLORS['accent_blue']
        )
        loading_label.pack(pady=100)

    def _display_results(self):
        """Afficher les résultats de simulation"""
        # Clear
        for widget in self.results_container.winfo_children():
            widget.destroy()

        if not self.current_result:
            return

        result = self.current_result

        # Section 1: Résumé global
        self._create_summary_section(result)

        # Section 2: Métriques de risque
        self._create_risk_metrics_section(result)

        # Section 3: Positions impactées
        self._create_positions_section(result)

        # Section 4: Recommandations
        self._create_recommendations_section(result)

    def _create_summary_section(self, result: Dict):
        """Section résumé global"""
        summary_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=COLORS['bg_tertiary'],
            corner_radius=10
        )
        summary_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            summary_frame,
            text=result.get('scenario_name', 'Simulation'),
            font=("Segoe UI", 16, "bold")
        )
        title.pack(pady=(15, 10))

        # Impact global
        impact = result.get('impact_percent', 0)
        impact_color = COLORS['accent_green'] if impact >= 0 else COLORS['accent_red']

        impact_label = ctk.CTkLabel(
            summary_frame,
            text=f"Impact Global: {impact:+.2f}%",
            font=("Segoe UI", 24, "bold"),
            text_color=impact_color
        )
        impact_label.pack(pady=10)

        # Valeurs avant/après
        value_frame = ctk.CTkFrame(summary_frame, fg_color="transparent")
        value_frame.pack(fill="x", padx=20, pady=(0, 15))

        value_before = result.get('portfolio_value_before', 0)
        value_after = result.get('portfolio_value_after', 0)

        ctk.CTkLabel(
            value_frame,
            text=f"Avant: ${value_before:,.2f}",
            font=("Segoe UI", 12),
            text_color=COLORS['text_secondary']
        ).pack(side="left", padx=10)

        ctk.CTkLabel(
            value_frame,
            text="→",
            font=("Segoe UI", 14)
        ).pack(side="left")

        ctk.CTkLabel(
            value_frame,
            text=f"Après: ${value_after:,.2f}",
            font=("Segoe UI", 12, "bold"),
            text_color=impact_color
        ).pack(side="left", padx=10)

    def _create_risk_metrics_section(self, result: Dict):
        """Section métriques de risque"""
        metrics = result.get('risk_metrics', {})
        if not metrics:
            return

        metrics_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=COLORS['bg_tertiary'],
            corner_radius=10
        )
        metrics_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            metrics_frame,
            text="📊 Métriques de Risque",
            font=("Segoe UI", 14, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(15, 10))

        # Grid de métriques
        metrics_grid = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=15, pady=(0, 15))

        metrics_to_show = [
            ("VaR 95%", metrics.get('var_95', 0), "%"),
            ("CVaR 95%", metrics.get('cvar_95', 0), "%"),
            ("Max Drawdown", metrics.get('max_drawdown', 0), "%"),
            ("Stress Score", metrics.get('stress_score', 0), "/100"),
            ("Recovery Time", metrics.get('recovery_time_days', 0), " jours")
        ]

        for i, (label, value, unit) in enumerate(metrics_to_show):
            metric_card = ctk.CTkFrame(metrics_grid, fg_color=COLORS['bg_secondary'])
            metric_card.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="ew")

            ctk.CTkLabel(
                metric_card,
                text=label,
                font=("Segoe UI", 10),
                text_color=COLORS['text_secondary']
            ).pack(pady=(10, 2))

            value_text = f"{value:.1f}{unit}" if isinstance(value, float) else f"{value}{unit}"
            ctk.CTkLabel(
                metric_card,
                text=value_text,
                font=("Segoe UI", 14, "bold")
            ).pack(pady=(0, 10))

        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)

    def _create_positions_section(self, result: Dict):
        """Section positions impactées"""
        positions = result.get('position_impacts', [])
        if not positions:
            return

        positions_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=COLORS['bg_tertiary'],
            corner_radius=10
        )
        positions_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            positions_frame,
            text=f"📈 Positions Impactées ({len(positions)})",
            font=("Segoe UI", 14, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(15, 10))

        # Top 5 positions les plus impactées
        sorted_positions = sorted(positions, key=lambda p: abs(p.get('impact_percent', 0)), reverse=True)[:5]

        for pos in sorted_positions:
            pos_card = ctk.CTkFrame(positions_frame, fg_color=COLORS['bg_secondary'])
            pos_card.pack(fill="x", padx=15, pady=5)

            ticker = pos.get('ticker', '?')
            impact = pos.get('impact_percent', 0)
            impact_color = COLORS['accent_green'] if impact >= 0 else COLORS['accent_red']

            pos_info = ctk.CTkFrame(pos_card, fg_color="transparent")
            pos_info.pack(fill="x", padx=10, pady=10)

            ctk.CTkLabel(
                pos_info,
                text=ticker,
                font=("Segoe UI", 12, "bold")
            ).pack(side="left", padx=(0, 10))

            ctk.CTkLabel(
                pos_info,
                text=f"{impact:+.2f}%",
                font=("Segoe UI", 12, "bold"),
                text_color=impact_color
            ).pack(side="right")

    def _create_recommendations_section(self, result: Dict):
        """Section recommandations"""
        recommendations = result.get('recommendations', [])
        if not recommendations:
            return

        reco_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=COLORS['bg_tertiary'],
            corner_radius=10
        )
        reco_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            reco_frame,
            text=f"💡 Recommandations ({len(recommendations)})",
            font=("Segoe UI", 14, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(15, 10))

        # Liste des recommandations
        for reco in recommendations:
            reco_card = ctk.CTkFrame(reco_frame, fg_color=COLORS['bg_secondary'])
            reco_card.pack(fill="x", padx=15, pady=5)

            # Priorité + Titre
            priority = reco.get('priority', 'medium')
            priority_emoji = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"

            title_label = ctk.CTkLabel(
                reco_card,
                text=f"{priority_emoji} {reco.get('title', '')}",
                font=("Segoe UI", 11, "bold"),
                anchor="w"
            )
            title_label.pack(fill="x", padx=10, pady=(10, 5))

            # Description
            desc_label = ctk.CTkLabel(
                reco_card,
                text=reco.get('description', ''),
                font=("Segoe UI", 10),
                text_color=COLORS['text_secondary'],
                anchor="w",
                wraplength=500
            )
            desc_label.pack(fill="x", padx=10, pady=(0, 10))

    def _show_error(self, error_msg: str):
        """Afficher une erreur"""
        # Clear
        for widget in self.results_container.winfo_children():
            widget.destroy()

        error_label = ctk.CTkLabel(
            self.results_container,
            text=f"❌ Erreur\n\n{error_msg}",
            font=("Segoe UI", 12),
            text_color=COLORS['accent_red']
        )
        error_label.pack(pady=100)
