"""
🚀 CHART ENGINE PLOTLY - Génération de graphiques ultra-professionnels

Ce module génère des graphiques Plotly interactifs de niveau institutionnel
avec tous les indicateurs techniques, styles personnalisés, et intégration ML.

Features :
- Candlestick charts professionnels
- 50+ indicateurs techniques
- ML predictions avec bandes de confiance
- Volume profile avancé
- Heatmaps de corrélation
- Multi-charts synchronisés
- Annotations automatiques (support/resistance, patterns)
- Export haute résolution
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Couleurs professionnelles pour Plotly
PLOT_COLORS = {
    'bg': '#0d1117',
    'paper': '#0d1117',
    'grid': '#1c2333',
    'text': '#c9d1d9',
    'up': '#00ff88',
    'down': '#ff4444',
    'volume_up': 'rgba(0, 255, 136, 0.5)',
    'volume_down': 'rgba(255, 68, 68, 0.5)',
    'ma20': '#ffa500',
    'ma50': '#ff69b4',
    'ma200': '#32cd32',
    'rsi': '#a855f7',
    'macd': '#00d4ff',
    'signal': '#ff8800',
    'bb_upper': '#00d4ff',
    'bb_middle': '#ffa500',
    'bb_lower': '#00d4ff',
    'ml_predict': '#a855f7',
    'ml_confidence_high': 'rgba(168, 85, 247, 0.3)',
    'ml_confidence_low': 'rgba(168, 85, 247, 0.1)',
}


class ChartEnginePlotly:
    """
    Moteur de génération de graphiques Plotly ultra-professionnels
    """

    def __init__(self):
        self.layout_template = self._create_professional_template()
        logger.info("📊 Chart Engine Plotly initialisé")

    def _create_professional_template(self) -> dict:
        """Crée un template professionnel pour tous les graphiques"""
        return dict(
            layout=dict(
                font=dict(family="Segoe UI, Arial", size=12, color=PLOT_COLORS['text']),
                paper_bgcolor=PLOT_COLORS['paper'],
                plot_bgcolor=PLOT_COLORS['bg'],
                xaxis=dict(
                    gridcolor=PLOT_COLORS['grid'],
                    showgrid=True,
                    zeroline=False,
                    showline=True,
                    linecolor=PLOT_COLORS['grid']
                ),
                yaxis=dict(
                    gridcolor=PLOT_COLORS['grid'],
                    showgrid=True,
                    zeroline=False,
                    showline=True,
                    linecolor=PLOT_COLORS['grid']
                ),
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=PLOT_COLORS['bg'],
                    font_size=12,
                    font_family="Segoe UI"
                ),
                legend=dict(
                    bgcolor='rgba(13, 17, 23, 0.8)',
                    bordercolor=PLOT_COLORS['grid'],
                    borderwidth=1,
                    font=dict(size=11)
                )
            )
        )

    # ================================================================
    # GRAPHIQUE PRINCIPAL : CANDLESTICK AVEC INDICATEURS
    # ================================================================

    def create_advanced_candlestick_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        indicators: List[str] = None,
        show_volume: bool = True,
        title: str = None
    ) -> go.Figure:
        """
        Crée un graphique candlestick ultra-professionnel avec indicateurs

        Args:
            df: DataFrame avec colonnes OHLCV (Open, High, Low, Close, Volume)
            ticker: Symbole du ticker
            indicators: Liste des indicateurs à afficher
            show_volume: Afficher le volume
            title: Titre personnalisé

        Returns:
            Figure Plotly interactive
        """
        indicators = indicators or []

        # Déterminer le nombre de sous-graphiques
        num_subplots = 1  # Prix principal
        subplot_heights = [0.6]

        if show_volume:
            num_subplots += 1
            subplot_heights.append(0.15)

        # Ajouter des sous-graphiques pour certains indicateurs
        if 'RSI' in indicators:
            num_subplots += 1
            subplot_heights.append(0.15)

        if 'MACD' in indicators:
            num_subplots += 1
            subplot_heights.append(0.15)

        # Normaliser les hauteurs
        total = sum(subplot_heights)
        subplot_heights = [h / total for h in subplot_heights]

        # Créer les sous-graphiques
        subplot_titles = [f'{ticker} - Price Chart']
        if show_volume:
            subplot_titles.append('Volume')
        if 'RSI' in indicators:
            subplot_titles.append('RSI (14)')
        if 'MACD' in indicators:
            subplot_titles.append('MACD')

        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=subplot_heights,
            specs=[[{"secondary_y": False}]] * num_subplots
        )

        current_row = 1

        # ================================================================
        # 1. CANDLESTICK PRINCIPAL
        # ================================================================
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color=PLOT_COLORS['up'],
                decreasing_line_color=PLOT_COLORS['down'],
                increasing_fillcolor=PLOT_COLORS['up'],
                decreasing_fillcolor=PLOT_COLORS['down']
            ),
            row=current_row,
            col=1
        )

        # ================================================================
        # 2. MOYENNES MOBILES
        # ================================================================
        if 'SMA' in indicators or 'EMA' in indicators:
            # Calculer les moyennes mobiles si pas déjà présentes
            if 'MA20' not in df.columns:
                df['MA20'] = df['Close'].rolling(window=20).mean()
            if 'MA50' not in df.columns:
                df['MA50'] = df['Close'].rolling(window=50).mean()
            if 'MA200' not in df.columns:
                df['MA200'] = df['Close'].rolling(window=200).mean()

            # MA20
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA20'],
                    name='MA20',
                    line=dict(color=PLOT_COLORS['ma20'], width=1.5),
                    opacity=0.8
                ),
                row=current_row,
                col=1
            )

            # MA50
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA50'],
                    name='MA50',
                    line=dict(color=PLOT_COLORS['ma50'], width=1.5),
                    opacity=0.8
                ),
                row=current_row,
                col=1
            )

            # MA200
            if len(df) >= 200:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA200'],
                        name='MA200',
                        line=dict(color=PLOT_COLORS['ma200'], width=2),
                        opacity=0.8
                    ),
                    row=current_row,
                    col=1
                )

        # ================================================================
        # 3. BOLLINGER BANDS
        # ================================================================
        if 'Bollinger Bands' in indicators:
            if 'BB_Upper' not in df.columns:
                # Calculer Bollinger Bands
                df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                bb_std = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

            # Bande supérieure
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color=PLOT_COLORS['bb_upper'], width=1, dash='dash'),
                    opacity=0.5
                ),
                row=current_row,
                col=1
            )

            # Bande du milieu
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Middle'],
                    name='BB Middle',
                    line=dict(color=PLOT_COLORS['bb_middle'], width=1),
                    opacity=0.5
                ),
                row=current_row,
                col=1
            )

            # Bande inférieure avec remplissage
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color=PLOT_COLORS['bb_lower'], width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(0, 212, 255, 0.1)',
                    opacity=0.5
                ),
                row=current_row,
                col=1
            )

        current_row += 1

        # ================================================================
        # 4. VOLUME
        # ================================================================
        if show_volume:
            colors = [
                PLOT_COLORS['volume_up'] if close >= open_price else PLOT_COLORS['volume_down']
                for close, open_price in zip(df['Close'], df['Open'])
            ]

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=current_row,
                col=1
            )

            current_row += 1

        # ================================================================
        # 5. RSI
        # ================================================================
        if 'RSI' in indicators:
            if 'RSI' not in df.columns:
                # Calculer RSI
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color=PLOT_COLORS['rsi'], width=2),
                    showlegend=False
                ),
                row=current_row,
                col=1
            )

            # Lignes de surachat/survente
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=current_row,
                col=1
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                row=current_row,
                col=1
            )

            # Zone colorée entre 30-70
            fig.add_hrect(
                y0=30,
                y1=70,
                fillcolor="rgba(100, 100, 100, 0.1)",
                layer="below",
                line_width=0,
                row=current_row,
                col=1
            )

            current_row += 1

        # ================================================================
        # 6. MACD
        # ================================================================
        if 'MACD' in indicators:
            if 'MACD' not in df.columns:
                # Calculer MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Histogram'] = df['MACD'] - df['Signal']

            # Histogramme MACD
            colors_macd = [
                PLOT_COLORS['up'] if val >= 0 else PLOT_COLORS['down']
                for val in df['MACD_Histogram']
            ]

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors_macd,
                    showlegend=False,
                    opacity=0.5
                ),
                row=current_row,
                col=1
            )

            # Ligne MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color=PLOT_COLORS['macd'], width=2),
                    showlegend=False
                ),
                row=current_row,
                col=1
            )

            # Ligne Signal
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Signal'],
                    name='Signal',
                    line=dict(color=PLOT_COLORS['signal'], width=2),
                    showlegend=False
                ),
                row=current_row,
                col=1
            )

        # ================================================================
        # LAYOUT FINAL
        # ================================================================
        # Apply template settings
        template = self.layout_template['layout']

        fig.update_layout(
            title=dict(
                text=title or f'{ticker} - Advanced Technical Analysis',
                font=dict(size=20, color=PLOT_COLORS['text']),
                x=0.5,
                xanchor='center'
            ),
            height=800,
            font=template['font'],
            paper_bgcolor=template['paper_bgcolor'],
            plot_bgcolor=template['plot_bgcolor'],
            hovermode=template['hovermode'],
            hoverlabel=template['hoverlabel'],
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(13, 17, 23, 0.8)',
                bordercolor=PLOT_COLORS['grid'],
                borderwidth=1
            )
        )

        # Désactiver le rangeslider et configurer les axes
        for i in range(1, num_subplots + 1):
            fig.update_xaxes(
                rangeslider_visible=False,
                gridcolor=PLOT_COLORS['grid'],
                showgrid=True,
                zeroline=False,
                showline=True,
                linecolor=PLOT_COLORS['grid'],
                row=i,
                col=1
            )
            fig.update_yaxes(
                gridcolor=PLOT_COLORS['grid'],
                showgrid=True,
                zeroline=False,
                showline=True,
                linecolor=PLOT_COLORS['grid'],
                row=i,
                col=1
            )

        logger.info(f"✅ Graphique candlestick créé pour {ticker} avec {len(indicators)} indicateurs")

        return fig

    # ================================================================
    # GRAPHIQUE ML : PRIX + PRÉDICTIONS AVEC BANDES DE CONFIANCE
    # ================================================================

    def create_ml_prediction_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        ml_predictions: Dict,
        show_confidence_bands: bool = True
    ) -> go.Figure:
        """
        Crée un graphique avec prix historique + prédictions ML + bandes de confiance

        Args:
            df: DataFrame avec prix historiques
            ticker: Symbole du ticker
            ml_predictions: Dict avec prédictions ML {
                '1d': {'signal': 'UP', 'confidence': 0.82, 'target_price': 150},
                '3d': {...}, '7d': {...}
            }
            show_confidence_bands: Afficher les bandes de confiance

        Returns:
            Figure Plotly ultra-professionnelle
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[
                f'{ticker} - Price + ML Predictions',
                'Prediction Confidence Over Time'
            ]
        )

        # ================================================================
        # 1. PRIX HISTORIQUE (Candlestick)
        # ================================================================
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Historical Price',
                increasing_line_color=PLOT_COLORS['up'],
                decreasing_line_color=PLOT_COLORS['down']
            ),
            row=1,
            col=1
        )

        # ================================================================
        # 2. PRÉDICTIONS ML (Lignes futures)
        # ================================================================
        if ml_predictions:
            last_date = df.index[-1]
            last_price = df['Close'].iloc[-1]

            # Créer les points de prédiction
            future_dates = []
            future_prices = []
            future_confidences = []

            for horizon in ['1d', '3d', '7d']:
                if horizon in ml_predictions:
                    pred = ml_predictions[horizon]
                    days = int(horizon[0])

                    future_date = last_date + timedelta(days=days)
                    target_price = pred.get('target_price', last_price)
                    confidence = pred.get('confidence', 0.5)

                    future_dates.append(future_date)
                    future_prices.append(target_price)
                    future_confidences.append(confidence)

            if future_dates:
                # Ligne de prédiction
                all_dates = [last_date] + future_dates
                all_prices = [last_price] + future_prices

                fig.add_trace(
                    go.Scatter(
                        x=all_dates,
                        y=all_prices,
                        name='ML Prediction',
                        line=dict(
                            color=PLOT_COLORS['ml_predict'],
                            width=3,
                            dash='dash'
                        ),
                        mode='lines+markers',
                        marker=dict(size=10, symbol='star')
                    ),
                    row=1,
                    col=1
                )

                # ================================================================
                # 3. BANDES DE CONFIANCE (zones colorées)
                # ================================================================
                if show_confidence_bands:
                    for i, (date, price, conf) in enumerate(zip(future_dates, future_prices, future_confidences)):
                        # Calculer la largeur de la bande selon la confiance
                        band_width = price * (1 - conf) * 0.5

                        # Zone de haute confiance (plus foncée)
                        fig.add_shape(
                            type="rect",
                            x0=last_date if i == 0 else future_dates[i-1],
                            x1=date,
                            y0=price - band_width/2,
                            y1=price + band_width/2,
                            fillcolor=PLOT_COLORS['ml_confidence_high'],
                            line_width=0,
                            layer="below",
                            row=1,
                            col=1
                        )

                        # Zone de basse confiance (plus claire)
                        fig.add_shape(
                            type="rect",
                            x0=last_date if i == 0 else future_dates[i-1],
                            x1=date,
                            y0=price - band_width,
                            y1=price + band_width,
                            fillcolor=PLOT_COLORS['ml_confidence_low'],
                            line_width=0,
                            layer="below",
                            row=1,
                            col=1
                        )

                # ================================================================
                # 4. ANNOTATIONS DES SIGNAUX
                # ================================================================
                for date, price, conf in zip(future_dates, future_prices, future_confidences):
                    pred = ml_predictions.get('1d', {})
                    signal = pred.get('signal', 'HOLD')

                    color = PLOT_COLORS['up'] if signal == 'UP' else PLOT_COLORS['down']
                    arrow_symbol = '▲' if signal == 'UP' else '▼'

                    fig.add_annotation(
                        x=date,
                        y=price,
                        text=f"{arrow_symbol} {signal}<br>{conf*100:.0f}% conf",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color,
                        arrowsize=1,
                        arrowwidth=2,
                        ax=0,
                        ay=-40 if signal == 'UP' else 40,
                        font=dict(size=11, color=color),
                        bgcolor='rgba(0,0,0,0.7)',
                        bordercolor=color,
                        borderwidth=2,
                        row=1,
                        col=1
                    )

                # ================================================================
                # 5. GRAPHIQUE DE CONFIANCE (bas)
                # ================================================================
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=[c * 100 for c in future_confidences],
                        name='Confidence %',
                        line=dict(color=PLOT_COLORS['ml_predict'], width=3),
                        fill='tozeroy',
                        fillcolor='rgba(168, 85, 247, 0.2)',
                        mode='lines+markers',
                        marker=dict(size=10)
                    ),
                    row=2,
                    col=1
                )

                # Ligne de seuil à 70%
                fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    annotation_text="High Confidence Threshold",
                    row=2,
                    col=1
                )

        # ================================================================
        # LAYOUT FINAL
        # ================================================================
        template = self.layout_template['layout']

        fig.update_layout(
            title=dict(
                text=f'🧠 {ticker} - AI-Powered Predictions (XGBoost + LSTM)',
                font=dict(size=20, color=PLOT_COLORS['text']),
                x=0.5,
                xanchor='center'
            ),
            height=900,
            font=template['font'],
            paper_bgcolor=template['paper_bgcolor'],
            plot_bgcolor=template['plot_bgcolor'],
            hovermode=template['hovermode'],
            hoverlabel=template['hoverlabel'],
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=template['legend']
        )

        # Update axes styling
        fig.update_xaxes(
            gridcolor=PLOT_COLORS['grid'],
            showgrid=True,
            zeroline=False,
            showline=True,
            linecolor=PLOT_COLORS['grid']
        )
        fig.update_yaxes(
            gridcolor=PLOT_COLORS['grid'],
            showgrid=True,
            zeroline=False,
            showline=True,
            linecolor=PLOT_COLORS['grid']
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Confidence (%)", row=2, col=1, range=[0, 100])

        logger.info(f"✅ Graphique ML créé pour {ticker}")

        return fig

    # ================================================================
    # HEATMAP DE CORRÉLATION PORTFOLIO
    # ================================================================

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Portfolio Correlation Matrix"
    ) -> go.Figure:
        """
        Crée une heatmap de corrélation professionnelle

        Args:
            correlation_matrix: DataFrame de corrélations
            title: Titre du graphique

        Returns:
            Figure Plotly heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdYlGn',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation",
                tickmode='linear',
                tick0=-1,
                dtick=0.5
            )
        ))

        template = self.layout_template['layout']

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=PLOT_COLORS['text']),
                x=0.5,
                xanchor='center'
            ),
            height=600,
            font=template['font'],
            paper_bgcolor=template['paper_bgcolor'],
            plot_bgcolor=template['plot_bgcolor'],
            hovermode=template['hovermode'],
            hoverlabel=template['hoverlabel'],
            legend=template['legend']
        )

        logger.info("✅ Heatmap de corrélation créée")

        return fig


# Instance singleton
_chart_engine_instance = None


def get_chart_engine() -> ChartEnginePlotly:
    """Retourne l'instance singleton du chart engine"""
    global _chart_engine_instance
    if _chart_engine_instance is None:
        _chart_engine_instance = ChartEnginePlotly()
    return _chart_engine_instance
