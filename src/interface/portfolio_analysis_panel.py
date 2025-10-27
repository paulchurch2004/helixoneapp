"""
Panel d'affichage des analyses de portfolio
Affiche les analyses automatiques (7h00 + 17h00) et permet d'en lancer manuellement
"""

import customtkinter as ctk
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading


class PortfolioAnalysisPanel(ctk.CTkScrollableFrame):
    """
    Panel d'affichage des analyses de portfolio

    Fonctionnalités:
    - Affichage de la dernière analyse (matin/soir)
    - Bouton pour lancer une analyse manuelle
    - Health score global du portfolio
    - Liste des alertes
    - Liste des recommandations
    """

    def __init__(self, master, api_client=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self.api_client = api_client
        self.current_analysis = None

        self._setup_ui()

    def _setup_ui(self):
        """Créer l'interface"""
        # Header avec titre et bouton refresh
        header_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        header_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            header_frame,
            text="📊 Analyse de Portfolio",
            font=("Segoe UI", 28, "bold")
        ).pack(side="left", padx=20, pady=15)

        # Bouton "Analyser Maintenant"
        self.analyze_button = ctk.CTkButton(
            header_frame,
            text="🔄 Analyser Maintenant",
            font=("Segoe UI", 14, "bold"),
            command=self._on_analyze_click,
            height=40,
            fg_color=("#2ecc71", "#27ae60"),
            hover_color=("#27ae60", "#229954")
        )
        self.analyze_button.pack(side="right", padx=20)

        # Info sur les analyses automatiques
        info_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        info_frame.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(
            info_frame,
            text="ℹ️  Analyses automatiques : 7h00 EST (matin) + 17h00 EST (soir)",
            font=("Segoe UI", 12),
            text_color="gray"
        ).pack(pady=10)

        # Container pour les résultats
        self.results_container = ctk.CTkFrame(self, fg_color="transparent")
        self.results_container.pack(fill="both", expand=True, padx=20)

        # Charger la dernière analyse au démarrage
        self._load_latest_analysis()

    def _on_analyze_click(self):
        """Lancer une analyse manuelle"""
        self.analyze_button.configure(state="disabled", text="⏳ Analyse en cours...")

        def run_analysis():
            try:
                # TODO: Appeler l'endpoint d'analyse portfolio manuelle
                # Pour l'instant, recharger la dernière analyse
                self._load_latest_analysis()

                self.after(0, lambda: self.analyze_button.configure(
                    state="normal",
                    text="🔄 Analyser Maintenant"
                ))

                # Toast notification
                self._show_toast("✅ Analyse terminée !")

            except Exception as e:
                self.after(0, lambda: self.analyze_button.configure(
                    state="normal",
                    text="🔄 Analyser Maintenant"
                ))
                self._show_toast(f"❌ Erreur : {str(e)}")

        # Lancer dans un thread
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

    def _load_latest_analysis(self):
        """Charger la dernière analyse depuis l'API"""
        def fetch():
            try:
                if not self.api_client:
                    self._display_no_data("Client API non initialisé")
                    return

                # Récupérer analyse
                analysis = self.api_client.get_portfolio_analysis()

                # Récupérer alertes
                alerts_data = self.api_client.get_portfolio_alerts()
                alerts = alerts_data.get("alerts", [])

                # Récupérer recommandations
                rec_data = self.api_client.get_portfolio_recommendations()
                recommendations = rec_data.get("recommendations", [])

                # Combiner
                full_data = {
                    "analysis": analysis,
                    "alerts": alerts,
                    "recommendations": recommendations
                }

                # Afficher dans le main thread
                self.after(0, lambda: self._display_analysis(full_data))

            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "No analysis found" in error_msg:
                    self.after(0, lambda: self._display_no_data(
                        "Aucune analyse disponible.\n\n"
                        "Les analyses automatiques sont effectuées à 7h00 et 17h00 EST.\n"
                        "Cliquez sur 'Analyser Maintenant' pour lancer une analyse manuelle."
                    ))
                else:
                    self.after(0, lambda: self._display_error(error_msg))

        # Lancer dans un thread
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()

    def _display_analysis(self, data: Dict):
        """Afficher l'analyse complète"""
        # Clear container
        for widget in self.results_container.winfo_children():
            widget.destroy()

        analysis = data.get("analysis", {})
        alerts = data.get("alerts", [])
        recommendations = data.get("recommendations", [])

        # Section 1: Health Score global
        self._create_health_section(analysis)

        # Section 2: Statistiques portfolio
        self._create_stats_section(analysis)

        # Section 3: Alertes
        self._create_alerts_section(alerts)

        # Section 4: Recommandations
        self._create_recommendations_section(recommendations)

    def _create_health_section(self, analysis: Dict):
        """Créer section health score"""
        health_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=("gray85", "gray20")
        )
        health_frame.pack(fill="x", pady=(0, 20))

        # Titre
        ctk.CTkLabel(
            health_frame,
            text="💊 Santé du Portfolio",
            font=("Segoe UI", 20, "bold")
        ).pack(pady=(15, 10))

        # Health score
        health_score = analysis.get("health_score", 0)

        if health_score >= 75:
            emoji = "🟢"
            status = "EXCELLENT"
            color = "#2ecc71"
        elif health_score >= 60:
            emoji = "🟡"
            status = "BON"
            color = "#f39c12"
        elif health_score >= 40:
            emoji = "🟠"
            status = "MOYEN"
            color = "#e67e22"
        else:
            emoji = "🔴"
            status = "RISQUÉ"
            color = "#e74c3c"

        ctk.CTkLabel(
            health_frame,
            text=f"{emoji} {health_score:.1f}/100",
            font=("Segoe UI", 48, "bold"),
            text_color=color
        ).pack(pady=10)

        ctk.CTkLabel(
            health_frame,
            text=status,
            font=("Segoe UI", 16, "bold"),
            text_color=color
        ).pack(pady=(0, 5))

        # Sentiment portfolio
        sentiment = analysis.get("portfolio_sentiment", "NEUTRAL")
        sentiment_emoji = {
            "BULLISH": "📈",
            "BEARISH": "📉",
            "NEUTRAL": "➡️"
        }.get(sentiment, "➡️")

        ctk.CTkLabel(
            health_frame,
            text=f"{sentiment_emoji} Sentiment: {sentiment}",
            font=("Segoe UI", 14),
            text_color="gray"
        ).pack(pady=(0, 15))

    def _create_stats_section(self, analysis: Dict):
        """Créer section statistiques"""
        stats_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=("gray85", "gray20")
        )
        stats_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            stats_frame,
            text="📈 Statistiques",
            font=("Segoe UI", 18, "bold")
        ).pack(pady=(15, 10), anchor="w", padx=15)

        # Grid de stats
        grid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        grid.pack(fill="x", padx=15, pady=(0, 15))

        stats = [
            ("Positions", analysis.get("num_positions", 0), "📦"),
            ("Retour Attendu 7j", f"{analysis.get('expected_return_7d', 0):.1f}%", "📊"),
            ("Risque Baissier", f"{analysis.get('downside_risk_pct', 0):.1f}%", "⚠️"),
            ("Alertes", analysis.get("num_alerts", 0), "🔔"),
            ("Alertes Critiques", analysis.get("num_critical_alerts", 0), "🔴"),
            ("Recommandations", analysis.get("num_recommendations", 0), "💡")
        ]

        for i, (label, value, emoji) in enumerate(stats):
            card = ctk.CTkFrame(grid, fg_color=("gray90", "gray17"))
            card.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky="ew")
            grid.columnconfigure(i % 3, weight=1)

            ctk.CTkLabel(
                card,
                text=emoji,
                font=("Segoe UI", 20)
            ).pack(pady=(10, 0))

            ctk.CTkLabel(
                card,
                text=str(value),
                font=("Segoe UI", 24, "bold")
            ).pack(pady=2)

            ctk.CTkLabel(
                card,
                text=label,
                font=("Segoe UI", 10),
                text_color="gray"
            ).pack(pady=(0, 10))

        # Timestamp
        timestamp = analysis.get("analysis_time", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%d/%m/%Y à %H:%M")
        except:
            time_str = timestamp

        ctk.CTkLabel(
            stats_frame,
            text=f"🕐 Dernière analyse: {time_str}",
            font=("Segoe UI", 10),
            text_color="gray"
        ).pack(pady=(0, 10), anchor="e", padx=15)

    def _create_alerts_section(self, alerts: list):
        """Créer section alertes"""
        alerts_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=("gray85", "gray20")
        )
        alerts_frame.pack(fill="x", pady=(0, 20))

        # Header
        header = ctk.CTkFrame(alerts_frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="🔔 Alertes Actives",
            font=("Segoe UI", 18, "bold")
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text=f"{len(alerts)} alerte{'s' if len(alerts) > 1 else ''}",
            font=("Segoe UI", 12),
            text_color="gray"
        ).pack(side="right")

        # Liste des alertes
        if not alerts:
            ctk.CTkLabel(
                alerts_frame,
                text="Aucune alerte active",
                font=("Segoe UI", 12),
                text_color="gray"
            ).pack(pady=(0, 15))
        else:
            alerts_list = ctk.CTkScrollableFrame(
                alerts_frame,
                height=200,
                fg_color=("gray90", "gray17")
            )
            alerts_list.pack(fill="both", padx=15, pady=(0, 15))

            for alert in alerts[:10]:  # Max 10 alertes
                self._create_alert_card(alerts_list, alert)

    def _create_alert_card(self, parent, alert: Dict):
        """Créer une card d'alerte"""
        severity = alert.get("severity", "INFO")
        ticker = alert.get("ticker", "N/A")
        title = alert.get("title", "")
        message = alert.get("message", "")

        # Couleur selon sévérité
        severity_config = {
            "CRITICAL": ("🔴", "#e74c3c"),
            "WARNING": ("🟠", "#f39c12"),
            "OPPORTUNITY": ("🟢", "#2ecc71"),
            "INFO": ("🔵", "#3498db")
        }
        emoji, color = severity_config.get(severity, ("🔵", "#3498db"))

        card = ctk.CTkFrame(parent, fg_color=("gray95", "gray15"))
        card.pack(fill="x", pady=5)

        # Header de l'alerte
        alert_header = ctk.CTkFrame(card, fg_color="transparent")
        alert_header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            alert_header,
            text=f"{emoji} {ticker}",
            font=("Segoe UI", 13, "bold"),
            text_color=color
        ).pack(side="left")

        ctk.CTkLabel(
            alert_header,
            text=severity,
            font=("Segoe UI", 10),
            text_color=color
        ).pack(side="right")

        # Titre
        ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            wraplength=600
        ).pack(fill="x", padx=10, pady=(0, 2))

        # Message
        ctk.CTkLabel(
            card,
            text=message,
            font=("Segoe UI", 10),
            text_color="gray",
            anchor="w",
            wraplength=600
        ).pack(fill="x", padx=10, pady=(0, 10))

    def _create_recommendations_section(self, recommendations: list):
        """Créer section recommandations"""
        rec_frame = ctk.CTkFrame(
            self.results_container,
            fg_color=("gray85", "gray20")
        )
        rec_frame.pack(fill="x", pady=(0, 20))

        # Header
        header = ctk.CTkFrame(rec_frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="💡 Recommandations",
            font=("Segoe UI", 18, "bold")
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text=f"{len(recommendations)} recommandation{'s' if len(recommendations) > 1 else ''}",
            font=("Segoe UI", 12),
            text_color="gray"
        ).pack(side="right")

        # Liste des recommandations
        if not recommendations:
            ctk.CTkLabel(
                rec_frame,
                text="Aucune recommandation disponible",
                font=("Segoe UI", 12),
                text_color="gray"
            ).pack(pady=(0, 15))
        else:
            rec_list = ctk.CTkScrollableFrame(
                rec_frame,
                height=250,
                fg_color=("gray90", "gray17")
            )
            rec_list.pack(fill="both", padx=15, pady=(0, 15))

            for rec in recommendations[:10]:  # Max 10 recommandations
                self._create_recommendation_card(rec_list, rec)

    def _create_recommendation_card(self, parent, rec: Dict):
        """Créer une card de recommandation"""
        ticker = rec.get("ticker", "N/A")
        action = rec.get("action", "HOLD")
        confidence = rec.get("confidence", 0)
        target_price = rec.get("target_price")
        stop_loss = rec.get("stop_loss")

        # Couleur selon action
        action_config = {
            "BUY": ("🟢 ACHAT", "#2ecc71"),
            "STRONG_BUY": ("🟢 ACHAT FORT", "#27ae60"),
            "HOLD": ("🟡 CONSERVER", "#f39c12"),
            "SELL": ("🔴 VENTE", "#e74c3c"),
            "STRONG_SELL": ("🔴 VENTE FORTE", "#c0392b")
        }
        action_text, color = action_config.get(action, ("🟡 CONSERVER", "#f39c12"))

        card = ctk.CTkFrame(parent, fg_color=("gray95", "gray15"))
        card.pack(fill="x", pady=5)

        # Header
        card_header = ctk.CTkFrame(card, fg_color="transparent")
        card_header.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            card_header,
            text=f"📊 {ticker}",
            font=("Segoe UI", 14, "bold")
        ).pack(side="left")

        ctk.CTkLabel(
            card_header,
            text=action_text,
            font=("Segoe UI", 12, "bold"),
            text_color=color
        ).pack(side="right")

        # Détails
        details_frame = ctk.CTkFrame(card, fg_color="transparent")
        details_frame.pack(fill="x", padx=10, pady=(0, 10))

        info_text = f"Confiance: {confidence:.0f}%"
        if target_price:
            info_text += f" | Cible: ${target_price:.2f}"
        if stop_loss:
            info_text += f" | Stop: ${stop_loss:.2f}"

        ctk.CTkLabel(
            details_frame,
            text=info_text,
            font=("Segoe UI", 10),
            text_color="gray"
        ).pack()

    def _display_no_data(self, message: str):
        """Afficher message quand pas de données"""
        for widget in self.results_container.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self.results_container, fg_color=("gray90", "gray17"))
        frame.pack(fill="both", expand=True, pady=50)

        ctk.CTkLabel(
            frame,
            text="📭",
            font=("Segoe UI", 64)
        ).pack(pady=(50, 20))

        ctk.CTkLabel(
            frame,
            text=message,
            font=("Segoe UI", 14),
            text_color="gray",
            justify="center"
        ).pack(pady=(0, 50))

    def _display_error(self, error_msg: str):
        """Afficher message d'erreur"""
        for widget in self.results_container.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self.results_container, fg_color=("gray90", "gray17"))
        frame.pack(fill="both", expand=True, pady=50)

        ctk.CTkLabel(
            frame,
            text="❌",
            font=("Segoe UI", 64)
        ).pack(pady=(50, 20))

        ctk.CTkLabel(
            frame,
            text="Erreur de chargement",
            font=("Segoe UI", 18, "bold"),
            text_color="#e74c3c"
        ).pack(pady=10)

        ctk.CTkLabel(
            frame,
            text=error_msg,
            font=("Segoe UI", 12),
            text_color="gray",
            wraplength=600
        ).pack(pady=(0, 50))

    def _show_toast(self, message: str):
        """Afficher une notification toast"""
        # Simple label temporaire (améliorer avec ToastNotifications si disponible)
        toast = ctk.CTkLabel(
            self,
            text=message,
            font=("Segoe UI", 12),
            fg_color=("#2ecc71", "#27ae60"),
            corner_radius=8
        )
        toast.place(relx=0.5, rely=0.95, anchor="center")

        # Disparaît après 3 secondes
        self.after(3000, toast.destroy)

    def refresh(self):
        """Rafraîchir l'affichage"""
        self._load_latest_analysis()
