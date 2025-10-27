"""
Composant d'affichage des résultats d'analyse ML
Affiche les prédictions ML, health score, et recommandations
"""

import customtkinter as ctk
from typing import Dict, Any, Optional
from datetime import datetime


class MLResultsDisplay(ctk.CTkFrame):
    """
    Panel d'affichage des résultats d'analyse ML

    Affiche:
    - Health Score animé (0-100)
    - Prédictions ML (1j, 3j, 7j)
    - Recommandation finale avec confiance
    - Scores FXI (5 dimensions)
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")

    def display_results(self, result: Dict[str, Any], ticker: str):
        """
        Afficher les résultats d'analyse ML

        Args:
            result: Résultat de l'endpoint /api/analysis/ml-enhanced
            ticker: Ticker analysé
        """
        # Clear existing widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Header avec ticker
        self._create_header(ticker, result)

        # Section 1: Health Score + Recommandation
        self._create_summary_section(result)

        # Section 2: Prédictions ML
        self._create_ml_predictions(result)

        # Section 3: Scores FXI
        self._create_fxi_scores(result)

        # Section 4: Détails et insights
        self._create_details_section(result)

    def _create_header(self, ticker: str, result: Dict):
        """Créer header avec ticker et timestamp"""
        header_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        header_frame.pack(fill="x", padx=20, pady=(10, 20))

        # Ticker
        ticker_label = ctk.CTkLabel(
            header_frame,
            text=f"📊 {ticker.upper()}",
            font=("Segoe UI", 32, "bold"),
            text_color=("#1f538d", "#3b8ed0")
        )
        ticker_label.pack(side="left", padx=20, pady=15)

        # Timestamp
        timestamp = result.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%d/%m/%Y %H:%M")
        except:
            time_str = timestamp

        time_label = ctk.CTkLabel(
            header_frame,
            text=f"🕐 {time_str}",
            font=("Segoe UI", 12),
            text_color="gray"
        )
        time_label.pack(side="right", padx=20)

    def _create_summary_section(self, result: Dict):
        """Créer section résumé (Health Score + Recommandation)"""
        summary_frame = ctk.CTkFrame(self, fg_color="transparent")
        summary_frame.pack(fill="x", padx=20, pady=10)

        # Health Score (gauche)
        health_frame = ctk.CTkFrame(summary_frame, fg_color=("gray90", "gray17"))
        health_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        health_score = result.get("health_score", 50)

        # Emoji selon score
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
            text="Health Score",
            font=("Segoe UI", 14),
            text_color="gray"
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            health_frame,
            text=f"{emoji} {health_score:.1f}/100",
            font=("Segoe UI", 36, "bold"),
            text_color=color
        ).pack(pady=5)

        ctk.CTkLabel(
            health_frame,
            text=status,
            font=("Segoe UI", 12, "bold"),
            text_color=color
        ).pack(pady=(0, 15))

        # Recommandation (droite)
        rec_frame = ctk.CTkFrame(summary_frame, fg_color=("gray90", "gray17"))
        rec_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        recommendation = result.get("recommendation_final", "HOLD").upper()
        confidence = result.get("confidence", 50)

        # Emoji et couleur selon recommandation
        if recommendation in ["BUY", "STRONG_BUY", "ACHETER"]:
            rec_emoji = "🟢"
            rec_text = "ACHAT"
            rec_color = "#2ecc71"
        elif recommendation in ["SELL", "STRONG_SELL", "VENDRE"]:
            rec_emoji = "🔴"
            rec_text = "VENTE"
            rec_color = "#e74c3c"
        else:
            rec_emoji = "🟡"
            rec_text = "CONSERVER"
            rec_color = "#f39c12"

        ctk.CTkLabel(
            rec_frame,
            text="Recommandation",
            font=("Segoe UI", 14),
            text_color="gray"
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            rec_frame,
            text=f"{rec_emoji} {rec_text}",
            font=("Segoe UI", 36, "bold"),
            text_color=rec_color
        ).pack(pady=5)

        ctk.CTkLabel(
            rec_frame,
            text=f"Confiance: {confidence:.0f}%",
            font=("Segoe UI", 12),
            text_color="gray"
        ).pack(pady=(0, 15))

    def _create_ml_predictions(self, result: Dict):
        """Créer section prédictions ML"""
        ml_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        ml_frame.pack(fill="x", padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            ml_frame,
            text="🤖 Prédictions ML (XGBoost + LSTM)",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        ).pack(fill="x", padx=15, pady=(15, 10))

        ml_pred = result.get("ml_predictions", {})
        signal = ml_pred.get("signal", "HOLD")
        signal_strength = ml_pred.get("signal_strength", 50)

        # Signal global
        signal_frame = ctk.CTkFrame(ml_frame, fg_color="transparent")
        signal_frame.pack(fill="x", padx=15, pady=(0, 10))

        # Emoji signal
        if signal == "BUY":
            signal_emoji = "📈"
            signal_color = "#2ecc71"
            signal_text = "Signal HAUSSIER"
        elif signal == "SELL":
            signal_emoji = "📉"
            signal_color = "#e74c3c"
            signal_text = "Signal BAISSIER"
        else:
            signal_emoji = "➡️"
            signal_color = "#f39c12"
            signal_text = "Signal NEUTRE"

        ctk.CTkLabel(
            signal_frame,
            text=f"{signal_emoji} {signal_text}",
            font=("Segoe UI", 16, "bold"),
            text_color=signal_color
        ).pack(side="left")

        ctk.CTkLabel(
            signal_frame,
            text=f"Force: {signal_strength:.0f}%",
            font=("Segoe UI", 14),
            text_color="gray"
        ).pack(side="right")

        # Grid des prédictions (1j, 3j, 7j)
        pred_grid = ctk.CTkFrame(ml_frame, fg_color="transparent")
        pred_grid.pack(fill="x", padx=15, pady=(0, 15))

        horizons = [
            ("1j", ml_pred.get("prediction_1d"), ml_pred.get("confidence_1d")),
            ("3j", ml_pred.get("prediction_3d"), ml_pred.get("confidence_3d")),
            ("7j", ml_pred.get("prediction_7d"), ml_pred.get("confidence_7d"))
        ]

        for i, (horizon, direction, conf) in enumerate(horizons):
            pred_card = ctk.CTkFrame(pred_grid, fg_color=("gray90", "gray17"))
            pred_card.grid(row=0, column=i, padx=5, sticky="ew")
            pred_grid.columnconfigure(i, weight=1)

            ctk.CTkLabel(
                pred_card,
                text=horizon,
                font=("Segoe UI", 12, "bold"),
                text_color="gray"
            ).pack(pady=(10, 2))

            if direction and conf:
                # Direction
                if direction == "UP":
                    dir_emoji = "⬆️"
                    dir_text = "HAUSSE"
                    dir_color = "#2ecc71"
                elif direction == "DOWN":
                    dir_emoji = "⬇️"
                    dir_text = "BAISSE"
                    dir_color = "#e74c3c"
                else:
                    dir_emoji = "↔️"
                    dir_text = "STABLE"
                    dir_color = "#95a5a6"

                ctk.CTkLabel(
                    pred_card,
                    text=f"{dir_emoji} {dir_text}",
                    font=("Segoe UI", 14, "bold"),
                    text_color=dir_color
                ).pack(pady=2)

                ctk.CTkLabel(
                    pred_card,
                    text=f"{conf:.0f}% confiance",
                    font=("Segoe UI", 10),
                    text_color="gray"
                ).pack(pady=(0, 10))
            else:
                ctk.CTkLabel(
                    pred_card,
                    text="N/A",
                    font=("Segoe UI", 14),
                    text_color="gray"
                ).pack(pady=(0, 10))

        # Model version
        model_version = ml_pred.get("model_version", "unknown")
        ctk.CTkLabel(
            ml_frame,
            text=f"Modèle: {model_version}",
            font=("Segoe UI", 10),
            text_color="gray"
        ).pack(anchor="e", padx=15, pady=(0, 10))

    def _create_fxi_scores(self, result: Dict):
        """Créer section scores FXI"""
        fxi_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        fxi_frame.pack(fill="x", padx=20, pady=10)

        # Title
        ctk.CTkLabel(
            fxi_frame,
            text="📊 Analyse FXI (5 Dimensions)",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        ).pack(fill="x", padx=15, pady=(15, 10))

        # Scores
        scores = [
            ("Technique", result.get("score_technique", 0), "📈"),
            ("Fondamental", result.get("score_fondamental", 0), "💼"),
            ("Sentiment", result.get("score_sentiment", 0), "💬"),
            ("Risque", result.get("score_risque", 0), "⚠️"),
            ("Macro", result.get("score_macro", 0), "🌍")
        ]

        for name, score, emoji in scores:
            score_row = ctk.CTkFrame(fxi_frame, fg_color="transparent")
            score_row.pack(fill="x", padx=15, pady=5)

            # Label
            label_frame = ctk.CTkFrame(score_row, fg_color="transparent", width=150)
            label_frame.pack(side="left", fill="y")
            label_frame.pack_propagate(False)

            ctk.CTkLabel(
                label_frame,
                text=f"{emoji} {name}",
                font=("Segoe UI", 13),
                anchor="w"
            ).pack(side="left", padx=5)

            # Progress bar
            progress_frame = ctk.CTkFrame(score_row, fg_color="transparent")
            progress_frame.pack(side="left", fill="both", expand=True, padx=10)

            progress = ctk.CTkProgressBar(
                progress_frame,
                width=300,
                height=20,
                progress_color=self._get_score_color(score)
            )
            progress.pack(fill="x")
            progress.set(score / 100)

            # Score value
            ctk.CTkLabel(
                score_row,
                text=f"{score:.0f}",
                font=("Segoe UI", 13, "bold"),
                width=40
            ).pack(side="right")

        # Score FXI global
        fxi_score = result.get("score_fxi", 0)
        global_frame = ctk.CTkFrame(fxi_frame, fg_color=("gray90", "gray17"))
        global_frame.pack(fill="x", padx=15, pady=(10, 15))

        ctk.CTkLabel(
            global_frame,
            text=f"Score FXI Global: {fxi_score:.1f}/100",
            font=("Segoe UI", 14, "bold"),
            text_color=self._get_score_color(fxi_score)
        ).pack(pady=10)

    def _create_details_section(self, result: Dict):
        """Créer section détails et insights"""
        details_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        details_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))

        # Title
        ctk.CTkLabel(
            details_frame,
            text="📝 Détails de l'Analyse",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        ).pack(fill="x", padx=15, pady=(15, 10))

        # Scrollable textbox
        textbox = ctk.CTkTextbox(
            details_frame,
            font=("Consolas", 11),
            wrap="word",
            height=200
        )
        textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Format details
        details_text = self._format_details(result)
        textbox.insert("1.0", details_text)
        textbox.configure(state="disabled")

    def _format_details(self, result: Dict) -> str:
        """Formater les détails en texte lisible"""
        lines = []

        lines.append("═" * 60)
        lines.append("  SYNTHÈSE DE L'ANALYSE")
        lines.append("═" * 60)
        lines.append("")

        # Recommandation finale
        rec = result.get("recommendation_final", "N/A")
        conf = result.get("confidence", 0)
        lines.append(f"🎯 Recommandation Finale : {rec} (Confiance: {conf:.0f}%)")
        lines.append("")

        # Health Score
        health = result.get("health_score", 0)
        lines.append(f"💊 Health Score Global  : {health:.1f}/100")
        lines.append("")

        # ML Predictions
        ml_pred = result.get("ml_predictions", {})
        if ml_pred:
            lines.append("─" * 60)
            lines.append("🤖 PRÉDICTIONS ML")
            lines.append("─" * 60)
            lines.append(f"Signal : {ml_pred.get('signal', 'N/A')}")
            lines.append(f"Force  : {ml_pred.get('signal_strength', 0):.0f}%")
            lines.append("")
            lines.append(f"1 jour  : {ml_pred.get('prediction_1d', 'N/A')} ({ml_pred.get('confidence_1d', 0):.0f}% conf)")
            lines.append(f"3 jours : {ml_pred.get('prediction_3d', 'N/A')} ({ml_pred.get('confidence_3d', 0):.0f}% conf)")
            lines.append(f"7 jours : {ml_pred.get('prediction_7d', 'N/A')} ({ml_pred.get('confidence_7d', 0):.0f}% conf)")
            lines.append("")

        # FXI Scores
        lines.append("─" * 60)
        lines.append("📊 SCORES FXI")
        lines.append("─" * 60)
        lines.append(f"Score Global     : {result.get('score_fxi', 0):.1f}/100")
        lines.append(f"Technique        : {result.get('score_technique', 0):.1f}/100")
        lines.append(f"Fondamental      : {result.get('score_fondamental', 0):.1f}/100")
        lines.append(f"Sentiment        : {result.get('score_sentiment', 0):.1f}/100")
        lines.append(f"Risque           : {result.get('score_risque', 0):.1f}/100")
        lines.append(f"Macroéconomique  : {result.get('score_macro', 0):.1f}/100")
        lines.append("")

        # Metadata
        lines.append("─" * 60)
        lines.append("ℹ️  MÉTADONNÉES")
        lines.append("─" * 60)
        exec_time = result.get("execution_time", 0)
        lines.append(f"Temps d'exécution : {exec_time:.2f}s")

        ml_gen = ml_pred.get("generated_at", "N/A")
        lines.append(f"ML généré         : {ml_gen}")

        model = ml_pred.get("model_version", "N/A")
        lines.append(f"Version modèle ML : {model}")

        lines.append("")

        # Detailed backend analysis
        details = result.get("details", {})
        if details and isinstance(details, dict):
            lines.append("═" * 60)
            lines.append("  ANALYSE DÉTAILLÉE - MOTEUR INTELLIGENT")
            lines.append("═" * 60)
            lines.append("")

            # Technical Analysis Details
            if "technical" in details or "technical_indicators" in details:
                lines.append("─" * 60)
                lines.append("📈 ANALYSE TECHNIQUE")
                lines.append("─" * 60)
                tech_data = details.get("technical", details.get("technical_indicators", {}))
                if isinstance(tech_data, dict):
                    for key, value in tech_data.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {key}: {value:.2f}")
                        else:
                            lines.append(f"  {key}: {value}")
                lines.append("")

            # Fundamental Analysis Details
            if "fundamental" in details or "fundamental_data" in details:
                lines.append("─" * 60)
                lines.append("💼 ANALYSE FONDAMENTALE")
                lines.append("─" * 60)
                fund_data = details.get("fundamental", details.get("fundamental_data", {}))
                if isinstance(fund_data, dict):
                    for key, value in fund_data.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {key}: {value:.2f}")
                        else:
                            lines.append(f"  {key}: {value}")
                lines.append("")

            # Macro Economic Data
            if "macro" in details or "macro_economic" in details:
                lines.append("─" * 60)
                lines.append("🌍 DONNÉES MACROÉCONOMIQUES")
                lines.append("─" * 60)
                macro_data = details.get("macro", details.get("macro_economic", {}))
                if isinstance(macro_data, dict):
                    for key, value in macro_data.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {key}: {value:.2f}")
                        else:
                            lines.append(f"  {key}: {value}")
                lines.append("")

            # Sentiment Analysis
            if "sentiment" in details or "sentiment_data" in details:
                lines.append("─" * 60)
                lines.append("💭 ANALYSE DE SENTIMENT")
                lines.append("─" * 60)
                sent_data = details.get("sentiment", details.get("sentiment_data", {}))
                if isinstance(sent_data, dict):
                    for key, value in sent_data.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {key}: {value:.2f}")
                        else:
                            lines.append(f"  {key}: {value}")
                lines.append("")

            # Risk Assessment
            if "risk" in details or "risk_assessment" in details:
                lines.append("─" * 60)
                lines.append("⚠️  ÉVALUATION DES RISQUES")
                lines.append("─" * 60)
                risk_data = details.get("risk", details.get("risk_assessment", {}))
                if isinstance(risk_data, dict):
                    for key, value in risk_data.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {key}: {value:.2f}")
                        else:
                            lines.append(f"  {key}: {value}")
                lines.append("")

            # Data Sources
            if "sources" in details or "data_sources" in details:
                lines.append("─" * 60)
                lines.append("📡 SOURCES DE DONNÉES")
                lines.append("─" * 60)
                sources = details.get("sources", details.get("data_sources", []))
                if isinstance(sources, list):
                    for source in sources:
                        lines.append(f"  • {source}")
                elif isinstance(sources, dict):
                    for key, value in sources.items():
                        lines.append(f"  • {key}: {value}")
                lines.append("")

            # Additional details (any other fields)
            remaining_keys = set(details.keys()) - {
                "technical", "technical_indicators",
                "fundamental", "fundamental_data",
                "macro", "macro_economic",
                "sentiment", "sentiment_data",
                "risk", "risk_assessment",
                "sources", "data_sources",
                "engine_version"
            }

            if remaining_keys:
                lines.append("─" * 60)
                lines.append("📋 INFORMATIONS SUPPLÉMENTAIRES")
                lines.append("─" * 60)
                for key in sorted(remaining_keys):
                    value = details[key]
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.2f}")
                    elif isinstance(value, str):
                        lines.append(f"  {key}: {value}")
                    elif isinstance(value, (list, dict)):
                        lines.append(f"  {key}: {len(value)} éléments")
                lines.append("")

        lines.append("═" * 60)

        return "\n".join(lines)

    def _get_score_color(self, score: float) -> str:
        """Obtenir couleur selon score"""
        if score >= 75:
            return "#2ecc71"  # Vert
        elif score >= 60:
            return "#3498db"  # Bleu
        elif score >= 40:
            return "#f39c12"  # Orange
        else:
            return "#e74c3c"  # Rouge
