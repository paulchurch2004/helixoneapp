"""
Composant d'affichage des résultats d'analyse COMPLÈTE (8 étapes)
Affiche tous les détails de l'analyse ultra-complète identique à celle exécutée 2x/jour
"""

import customtkinter as ctk
from typing import Dict, Any, List
from datetime import datetime


class DeepAnalysisDisplay(ctk.CTkScrollableFrame):
    """
    Panel d'affichage des résultats d'analyse COMPLÈTE

    Affiche les 8 étapes:
    1. Executive Summary
    2. Health Score + Recommandation
    3. ML Predictions (XGBoost + LSTM)
    4. Alerts (Critical/Important/Info/Opportunity)
    5. Sentiment Analysis (trend, velocity, patterns)
    6. Data Collection (35+ sources)
    7. Upcoming Economic Events
    8. Position Analysis détaillée
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")

    def display_results(self, result: Dict[str, Any], ticker: str):
        """
        Afficher les résultats d'analyse complète

        Args:
            result: Résultat de l'endpoint /api/analysis/stock-deep-analysis
            ticker: Ticker analysé
        """
        # Clear existing widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Header avec ticker et indicateur "Analyse Complète"
        self._create_header(ticker, result)

        # Section 1: Executive Summary
        self._create_executive_summary(result)

        # Section 2: Health Score + Recommandation (comme l'ancien mais amélioré)
        self._create_summary_section(result)

        # Section 3: Alerts (Critical/Important/Info)
        self._create_alerts_section(result)

        # Section 4: ML Predictions Enhanced (1d/3d/7d)
        self._create_ml_predictions(result)

        # Section 5: Sentiment Analysis
        self._create_sentiment_section(result)

        # Section 6: Upcoming Economic Events
        self._create_events_section(result)

        # Section 7: Data Collection Stats
        self._create_data_sources_section(result)

        # Section 8: Position Analysis Details
        self._create_position_details(result)

    def _create_header(self, ticker: str, result: Dict):
        """Créer header avec ticker et badge 'ANALYSE COMPLÈTE'"""
        header_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        header_frame.pack(fill="x", padx=10, pady=(5, 15))

        # Ticker
        ticker_label = ctk.CTkLabel(
            header_frame,
            text=f"📊 {ticker.upper()}",
            font=("Segoe UI", 32, "bold"),
            text_color=("#1f538d", "#3b8ed0")
        )
        ticker_label.pack(side="left", padx=20, pady=15)

        # Badge "ANALYSE COMPLÈTE 8 ÉTAPES"
        badge = ctk.CTkFrame(header_frame, fg_color=("#2ecc71", "#27ae60"))
        badge.pack(side="left", padx=10)

        badge_label = ctk.CTkLabel(
            badge,
            text="✨ ANALYSE COMPLÈTE 8 ÉTAPES",
            font=("Segoe UI", 11, "bold"),
            text_color="white"
        )
        badge_label.pack(padx=15, pady=5)

        # Timestamp
        timestamp = result.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%d/%m/%Y %H:%M")
        except:
            time_str = "Maintenant"

        time_label = ctk.CTkLabel(
            header_frame,
            text=f"🕐 {time_str}",
            font=("Segoe UI", 12),
            text_color="gray"
        )
        time_label.pack(side="right", padx=20)

    def _create_executive_summary(self, result: Dict):
        """Créer section Executive Summary"""
        summary_text = result.get("executive_summary", "")
        if not summary_text:
            return

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            section_frame,
            text="📋 RÉSUMÉ EXÉCUTIF",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 5))

        # Texte du résumé
        summary_label = ctk.CTkTextbox(
            section_frame,
            height=100,
            font=("Segoe UI", 12),
            wrap="word"
        )
        summary_label.pack(fill="x", padx=15, pady=(5, 15))
        summary_label.insert("1.0", summary_text)
        summary_label.configure(state="disabled")

    def _create_summary_section(self, result: Dict):
        """Créer section Health Score + Recommandation"""
        summary_frame = ctk.CTkFrame(self, fg_color="transparent")
        summary_frame.pack(fill="x", padx=10, pady=10)

        # Health Score (gauche)
        health_frame = ctk.CTkFrame(summary_frame, fg_color=("gray90", "gray17"))
        health_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        health_score = result.get("health_score", 50)

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
            status = "FAIBLE"
            color = "#e74c3c"

        health_title = ctk.CTkLabel(
            health_frame,
            text="HEALTH SCORE",
            font=("Segoe UI", 12, "bold"),
            text_color="gray"
        )
        health_title.pack(pady=(15, 5))

        health_value = ctk.CTkLabel(
            health_frame,
            text=f"{emoji} {health_score:.0f}/100",
            font=("Segoe UI", 36, "bold"),
            text_color=color
        )
        health_value.pack(pady=5)

        health_status = ctk.CTkLabel(
            health_frame,
            text=status,
            font=("Segoe UI", 14, "bold"),
            text_color=color
        )
        health_status.pack(pady=(0, 15))

        # Recommandation (droite)
        rec_frame = ctk.CTkFrame(summary_frame, fg_color=("gray90", "gray17"))
        rec_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        recommendation = result.get("recommendation_final", "HOLD")
        confidence = result.get("confidence", 50)

        if recommendation in ["STRONG_BUY", "BUY", "ACHETER"]:
            rec_emoji = "🚀"
            rec_color = "#2ecc71"
            rec_text = "ACHETER"
        elif recommendation in ["STRONG_SELL", "SELL", "VENDRE"]:
            rec_emoji = "⚠️"
            rec_color = "#e74c3c"
            rec_text = "VENDRE"
        else:
            rec_emoji = "⏸️"
            rec_color = "#95a5a6"
            rec_text = "CONSERVER"

        rec_title = ctk.CTkLabel(
            rec_frame,
            text="RECOMMANDATION",
            font=("Segoe UI", 12, "bold"),
            text_color="gray"
        )
        rec_title.pack(pady=(15, 5))

        rec_value = ctk.CTkLabel(
            rec_frame,
            text=f"{rec_emoji} {rec_text}",
            font=("Segoe UI", 28, "bold"),
            text_color=rec_color
        )
        rec_value.pack(pady=5)

        confidence_label = ctk.CTkLabel(
            rec_frame,
            text=f"Confiance: {confidence:.0f}%",
            font=("Segoe UI", 14),
            text_color="gray"
        )
        confidence_label.pack(pady=(0, 15))

    def _create_alerts_section(self, result: Dict):
        """Créer section Alerts"""
        alerts = result.get("alerts", {})
        critical = alerts.get("critical", [])
        important = alerts.get("important", [])
        info = alerts.get("info", [])
        opportunity = alerts.get("opportunity", [])

        # Ne pas afficher la section si aucune alerte
        if not (critical or important or info or opportunity):
            return

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            section_frame,
            text="🚨 ALERTES",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 5))

        # Afficher chaque catégorie
        if critical:
            self._create_alert_category(section_frame, "CRITIQUE", critical, "#e74c3c", "🔴")
        if important:
            self._create_alert_category(section_frame, "IMPORTANT", important, "#e67e22", "🟠")
        if opportunity:
            self._create_alert_category(section_frame, "OPPORTUNITÉ", opportunity, "#2ecc71", "🟢")
        if info:
            self._create_alert_category(section_frame, "INFO", info, "#3498db", "ℹ️")

    def _create_alert_category(self, parent, category_name: str, alerts: List[Dict], color: str, emoji: str):
        """Créer une catégorie d'alertes"""
        category_frame = ctk.CTkFrame(parent, fg_color="transparent")
        category_frame.pack(fill="x", padx=10, pady=5)

        # Titre de catégorie
        cat_title = ctk.CTkLabel(
            category_frame,
            text=f"{emoji} {category_name} ({len(alerts)})",
            font=("Segoe UI", 12, "bold"),
            text_color=color,
            anchor="w"
        )
        cat_title.pack(fill="x", padx=5, pady=(5, 2))

        # Liste des alertes
        for alert in alerts[:3]:  # Limiter à 3 alertes par catégorie
            alert_text = alert.get("message", alert.get("title", "Alerte"))
            alert_label = ctk.CTkLabel(
                category_frame,
                text=f"  • {alert_text}",
                font=("Segoe UI", 11),
                anchor="w",
                wraplength=700
            )
            alert_label.pack(fill="x", padx=15, pady=2)

    def _create_ml_predictions(self, result: Dict):
        """Créer section ML Predictions (XGBoost + LSTM)"""
        ml_preds = result.get("ml_predictions", {})

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        model_version = ml_preds.get("model_version", "XGBoost+LSTM")
        title = ctk.CTkLabel(
            section_frame,
            text=f"🧠 PRÉDICTIONS ML ({model_version})",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 10))

        # Signal global
        signal = ml_preds.get("signal", "HOLD")
        signal_strength = ml_preds.get("signal_strength", 50)

        if signal == "BUY":
            signal_emoji = "📈"
            signal_color = "#2ecc71"
            signal_text = "SIGNAL ACHAT"
        elif signal == "SELL":
            signal_emoji = "📉"
            signal_color = "#e74c3c"
            signal_text = "SIGNAL VENTE"
        else:
            signal_emoji = "➡️"
            signal_color = "#95a5a6"
            signal_text = "SIGNAL NEUTRE"

        signal_frame = ctk.CTkFrame(section_frame, fg_color=("gray85", "gray20"))
        signal_frame.pack(fill="x", padx=15, pady=(0, 10))

        signal_label = ctk.CTkLabel(
            signal_frame,
            text=f"{signal_emoji} {signal_text}",
            font=("Segoe UI", 18, "bold"),
            text_color=signal_color
        )
        signal_label.pack(side="left", padx=15, pady=10)

        strength_label = ctk.CTkLabel(
            signal_frame,
            text=f"Force: {signal_strength:.0f}%",
            font=("Segoe UI", 14),
            text_color="gray"
        )
        strength_label.pack(side="left", padx=10)

        # Prédictions 1j, 3j, 7j
        pred_container = ctk.CTkFrame(section_frame, fg_color="transparent")
        pred_container.pack(fill="x", padx=15, pady=(0, 15))

        for horizon, label_text in [("1d", "1 JOUR"), ("3d", "3 JOURS"), ("7d", "7 JOURS")]:
            pred = ml_preds.get(f"prediction_{horizon}", {})
            direction = pred if isinstance(pred, str) else pred.get("direction", "N/A")
            confidence = pred.get("confidence", 0) if isinstance(pred, dict) else 0

            pred_frame = ctk.CTkFrame(pred_container, fg_color=("gray85", "gray20"))
            pred_frame.pack(side="left", fill="both", expand=True, padx=5)

            label = ctk.CTkLabel(
                pred_frame,
                text=label_text,
                font=("Segoe UI", 10, "bold"),
                text_color="gray"
            )
            label.pack(pady=(10, 2))

            dir_label = ctk.CTkLabel(
                pred_frame,
                text=direction,
                font=("Segoe UI", 16, "bold")
            )
            dir_label.pack(pady=2)

            conf_label = ctk.CTkLabel(
                pred_frame,
                text=f"{confidence:.0f}%",
                font=("Segoe UI", 12),
                text_color="gray"
            )
            conf_label.pack(pady=(0, 10))

    def _create_sentiment_section(self, result: Dict):
        """Créer section Sentiment Analysis"""
        sentiment = result.get("sentiment_analysis", {})
        if not sentiment:
            return

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            section_frame,
            text="💭 ANALYSE SENTIMENT",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 10))

        # Conteneur horizontal
        container = ctk.CTkFrame(section_frame, fg_color="transparent")
        container.pack(fill="x", padx=15, pady=(0, 15))

        # Sentiment score
        sentiment_score = sentiment.get("sentiment_score", 50)
        if sentiment_score >= 70:
            sent_emoji = "😊"
            sent_color = "#2ecc71"
            sent_text = "POSITIF"
        elif sentiment_score >= 40:
            sent_emoji = "😐"
            sent_color = "#f39c12"
            sent_text = "NEUTRE"
        else:
            sent_emoji = "😟"
            sent_color = "#e74c3c"
            sent_text = "NÉGATIF"

        score_frame = ctk.CTkFrame(container, fg_color=("gray85", "gray20"))
        score_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        score_label = ctk.CTkLabel(
            score_frame,
            text=f"{sent_emoji} {sent_text}",
            font=("Segoe UI", 18, "bold"),
            text_color=sent_color
        )
        score_label.pack(pady=10)

        score_value = ctk.CTkLabel(
            score_frame,
            text=f"{sentiment_score:.0f}/100",
            font=("Segoe UI", 14),
            text_color="gray"
        )
        score_value.pack(pady=(0, 10))

        # Trend
        trend = sentiment.get("trend", "stable")
        if trend == "rising":
            trend_emoji = "📈"
            trend_text = "EN HAUSSE"
            trend_color = "#2ecc71"
        elif trend == "falling":
            trend_emoji = "📉"
            trend_text = "EN BAISSE"
            trend_color = "#e74c3c"
        else:
            trend_emoji = "➡️"
            trend_text = "STABLE"
            trend_color = "#95a5a6"

        trend_frame = ctk.CTkFrame(container, fg_color=("gray85", "gray20"))
        trend_frame.pack(side="left", fill="both", expand=True, padx=5)

        trend_label = ctk.CTkLabel(
            trend_frame,
            text=f"{trend_emoji} {trend_text}",
            font=("Segoe UI", 18, "bold"),
            text_color=trend_color
        )
        trend_label.pack(pady=10)

        velocity = sentiment.get("velocity", 0)
        velocity_label = ctk.CTkLabel(
            trend_frame,
            text=f"Vélocité: {velocity:+.1f}",
            font=("Segoe UI", 12),
            text_color="gray"
        )
        velocity_label.pack(pady=(0, 10))

    def _create_events_section(self, result: Dict):
        """Créer section Upcoming Economic Events"""
        events = result.get("upcoming_events", [])
        if not events:
            return

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            section_frame,
            text=f"📅 ÉVÉNEMENTS À VENIR ({len(events)})",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 10))

        # Liste des événements (max 5)
        for event in events[:5]:
            event_frame = ctk.CTkFrame(section_frame, fg_color=("gray85", "gray20"))
            event_frame.pack(fill="x", padx=15, pady=5)

            event_name = event.get("name", event.get("title", "Événement"))
            event_date = event.get("date", "")
            event_impact = event.get("impact", "medium")

            if event_impact == "high":
                impact_emoji = "🔴"
            elif event_impact == "medium":
                impact_emoji = "🟠"
            else:
                impact_emoji = "🟡"

            event_label = ctk.CTkLabel(
                event_frame,
                text=f"{impact_emoji} {event_name}",
                font=("Segoe UI", 12, "bold"),
                anchor="w"
            )
            event_label.pack(side="left", padx=10, pady=8)

            if event_date:
                date_label = ctk.CTkLabel(
                    event_frame,
                    text=event_date,
                    font=("Segoe UI", 10),
                    text_color="gray"
                )
                date_label.pack(side="right", padx=10)

    def _create_data_sources_section(self, result: Dict):
        """Créer section Data Collection (35+ sources)"""
        data_collection = result.get("data_collection", {})
        if not data_collection:
            return

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        sources_count = data_collection.get("sources_count", 35)
        title = ctk.CTkLabel(
            section_frame,
            text=f"📡 SOURCES DE DONNÉES ({sources_count}+ sources)",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 10))

        # Indicateurs de disponibilité
        sources_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        sources_frame.pack(fill="x", padx=15, pady=(0, 15))

        source_categories = [
            ("Social Media", ["reddit", "stocktwits"], "💬"),
            ("News", ["newsapi", "google_news"], "📰"),
            ("Financial Data", ["alpha_vantage", "finnhub", "yfinance"], "💹"),
            ("Macro Data", ["fred", "google_trends"], "📊"),
            ("Fundamentals", ["sec_edgar", "fmp"], "📈")
        ]

        for cat_name, source_keys, emoji in source_categories:
            available = any(data_collection.get(key, {}).get("available", False) for key in source_keys)

            cat_frame = ctk.CTkFrame(sources_frame, fg_color=("gray85", "gray20"))
            cat_frame.pack(side="left", fill="both", expand=True, padx=3)

            status_emoji = "✅" if available else "❌"
            cat_label = ctk.CTkLabel(
                cat_frame,
                text=f"{emoji} {cat_name}\n{status_emoji}",
                font=("Segoe UI", 10),
                justify="center"
            )
            cat_label.pack(pady=8, padx=5)

    def _create_position_details(self, result: Dict):
        """Créer section Position Analysis Details"""
        position = result.get("position_analysis", {})
        if not position:
            return

        section_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        section_frame.pack(fill="x", padx=10, pady=10)

        # Titre
        title = ctk.CTkLabel(
            section_frame,
            text="📊 ANALYSE DE POSITION",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        title.pack(fill="x", padx=15, pady=(10, 10))

        # Grille de métriques
        metrics_container = ctk.CTkFrame(section_frame, fg_color="transparent")
        metrics_container.pack(fill="x", padx=15, pady=(0, 15))

        # Ligne 1: Technical Score, Fundamental Score
        row1 = ctk.CTkFrame(metrics_container, fg_color="transparent")
        row1.pack(fill="x", pady=5)

        self._create_metric_box(row1, "TECHNIQUE", result.get("score_technique", 0), "/100")
        self._create_metric_box(row1, "FONDAMENTAL", result.get("score_fondamental", 0), "/100")

        # Ligne 2: Risk Score, Sentiment Score
        row2 = ctk.CTkFrame(metrics_container, fg_color="transparent")
        row2.pack(fill="x", pady=5)

        self._create_metric_box(row2, "RISQUE", result.get("score_risque", 0), "/100")
        self._create_metric_box(row2, "SENTIMENT", result.get("score_sentiment", 0), "/100")

    def _create_metric_box(self, parent, label: str, value: float, suffix: str):
        """Créer une box de métrique"""
        box = ctk.CTkFrame(parent, fg_color=("gray85", "gray20"))
        box.pack(side="left", fill="both", expand=True, padx=5)

        label_widget = ctk.CTkLabel(
            box,
            text=label,
            font=("Segoe UI", 10, "bold"),
            text_color="gray"
        )
        label_widget.pack(pady=(10, 2))

        value_widget = ctk.CTkLabel(
            box,
            text=f"{value:.0f}{suffix}",
            font=("Segoe UI", 20, "bold")
        )
        value_widget.pack(pady=(0, 10))
