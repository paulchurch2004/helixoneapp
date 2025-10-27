"""
Visualization - Dashboards Plotly pour résultats ML

Graphiques:
1. Equity curve vs benchmark
2. Drawdown chart
3. Returns distribution
4. Monte Carlo fan chart
5. Feature importance
6. Confusion matrix
7. Signals timeline

Output: HTML interactif ou serveur Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MLVisualizer:
    """
    Créateur de visualisations pour résultats ML
    """

    def __init__(self, template: str = 'plotly_dark'):
        """
        Args:
            template: Template Plotly ('plotly_dark', 'plotly_white', etc.)
        """
        self.template = template
        logger.info(f"MLVisualizer initialisé (template={template})")

    def plot_equity_curve(
        self,
        equity: pd.Series,
        benchmark: Optional[pd.Series] = None,
        signals: Optional[pd.DataFrame] = None,
        title: str = "Equity Curve"
    ) -> go.Figure:
        """
        Plot equity curve vs benchmark

        Args:
            equity: Série de valeurs du portfolio
            benchmark: Série de benchmark (optionnel)
            signals: DataFrame avec colonnes [date, action] (optionnel)
            title: Titre du graphique

        Returns:
            Figure Plotly
        """
        fig = go.Figure()

        # Equity curve
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            name='Strategy',
            line=dict(color='#00d4aa', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))

        # Benchmark
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                name='Benchmark',
                line=dict(color='#ffa600', width=2, dash='dash')
            ))

        # Signaux (BUY/SELL markers)
        if signals is not None and len(signals) > 0:
            buys = signals[signals['action'] == 'BUY']
            sells = signals[signals['action'] == 'SELL']

            if len(buys) > 0:
                buy_prices = [equity.loc[date] for date in buys['date']]
                fig.add_trace(go.Scatter(
                    x=buys['date'],
                    y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))

            if len(sells) > 0:
                sell_prices = [equity.loc[date] for date in sells['date']]
                fig.add_trace(go.Scatter(
                    x=sells['date'],
                    y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template=self.template,
            hovermode='x unified',
            height=600
        )

        return fig

    def plot_drawdown(
        self,
        equity: pd.Series,
        title: str = "Drawdown"
    ) -> go.Figure:
        """
        Plot drawdown chart

        Args:
            equity: Série de valeurs du portfolio
            title: Titre

        Returns:
            Figure Plotly
        """
        # Calculer drawdown
        cumulative = equity / equity.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = go.Figure()

        # Drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1)
        ))

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.template,
            hovermode='x unified',
            height=400
        )

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """
        Plot distribution des returns

        Args:
            returns: Série de returns
            title: Titre

        Returns:
            Figure Plotly
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Histogram", "Q-Q Plot"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                name='Returns',
                nbinsx=50,
                marker_color='#00d4aa'
            ),
            row=1, col=1
        )

        # Q-Q Plot (vs normal distribution)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='#ffa600')
            ),
            row=1, col=2
        )

        # Ligne théorique (distribution normale parfaite)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=theoretical_quantiles,
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )

        # Layout
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

        fig.update_layout(
            title=title,
            template=self.template,
            height=500,
            showlegend=True
        )

        return fig

    def plot_monte_carlo_fan(
        self,
        scenarios: np.ndarray,
        percentiles: List[int] = [5, 25, 50, 75, 95],
        title: str = "Monte Carlo Simulation"
    ) -> go.Figure:
        """
        Plot Monte Carlo fan chart

        Args:
            scenarios: Array (n_simulations, n_days)
            percentiles: Percentiles à afficher
            title: Titre

        Returns:
            Figure Plotly
        """
        n_days = scenarios.shape[1]
        dates = list(range(n_days))

        fig = go.Figure()

        # Percentiles
        colors = {
            5: 'rgba(255, 0, 0, 0.2)',
            25: 'rgba(255, 100, 0, 0.2)',
            50: 'rgba(0, 212, 170, 0.5)',
            75: 'rgba(100, 200, 100, 0.2)',
            95: 'rgba(0, 255, 0, 0.2)'
        }

        for p in sorted(percentiles):
            p_values = np.percentile(scenarios, p, axis=0)

            fig.add_trace(go.Scatter(
                x=dates,
                y=p_values,
                name=f'P{p}',
                line=dict(width=1 if p != 50 else 3),
                fillcolor=colors.get(p, 'rgba(100, 100, 100, 0.1)'),
                fill='tonexty' if p > min(percentiles) else None
            ))

        # Quelques trajectoires individuelles (sample)
        n_sample = min(100, scenarios.shape[0])
        sample_indices = np.random.choice(scenarios.shape[0], n_sample, replace=False)

        for idx in sample_indices[:10]:  # Afficher seulement 10 pour lisibilité
            fig.add_trace(go.Scatter(
                x=dates,
                y=scenarios[idx],
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.1)', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)",
            template=self.template,
            hovermode='x unified',
            height=600
        )

        return fig

    def plot_feature_importance(
        self,
        importances: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """
        Plot feature importance

        Args:
            importances: DataFrame avec colonnes [feature, importance]
            top_n: Nombre de top features à afficher
            title: Titre

        Returns:
            Figure Plotly
        """
        # Top N features
        top_features = importances.head(top_n).sort_values('importance')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True
            )
        ))

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            template=self.template,
            height=max(400, top_n * 20),
            showlegend=False
        )

        return fig

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str] = ['DOWN', 'FLAT', 'UP'],
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """
        Plot confusion matrix

        Args:
            confusion_matrix: Array 2D
            labels: Labels des classes
            title: Titre

        Returns:
            Figure Plotly
        """
        # Normaliser par ligne (% par classe réelle)
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template=self.template,
            height=500
        )

        return fig

    def create_dashboard(
        self,
        equity: pd.Series,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        signals: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict] = None,
        feature_importance: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Crée un dashboard complet

        Args:
            equity: Equity curve
            returns: Returns
            benchmark: Benchmark (optionnel)
            signals: Signaux (optionnel)
            metrics: Métriques (optionnel)
            feature_importance: Feature importance (optionnel)

        Returns:
            Figure Plotly avec subplots
        """
        # Créer subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Equity Curve",
                "Returns Distribution",
                "Drawdown",
                "Monthly Returns Heatmap",
                "Feature Importance" if feature_importance is not None else "Metrics",
                "Rolling Sharpe (90d)"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Equity curve
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name='Strategy', line=dict(color='#00d4aa', width=2)),
            row=1, col=1
        )

        if benchmark is not None:
            fig.add_trace(
                go.Scatter(x=benchmark.index, y=benchmark.values, name='Benchmark',
                           line=dict(color='#ffa600', width=2, dash='dash')),
                row=1, col=1
            )

        # 2. Returns distribution
        fig.add_trace(
            go.Histogram(x=returns * 100, name='Returns', nbinsx=50, marker_color='#00d4aa'),
            row=1, col=2
        )

        # 3. Drawdown
        cumulative = equity / equity.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, name='Drawdown',
                       fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=1)),
            row=2, col=1
        )

        # 4. Monthly returns heatmap
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        years = monthly_returns.index.year.unique()
        months = range(1, 13)

        heatmap_data = np.full((len(years), 12), np.nan)
        for i, year in enumerate(years):
            year_data = monthly_returns[monthly_returns.index.year == year]
            for month_val in year_data.index.month:
                month_idx = month_val - 1
                heatmap_data[i, month_idx] = year_data[year_data.index.month == month_val].values[0]

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=[str(y) for y in years],
                colorscale='RdYlGn',
                zmid=0,
                showscale=True
            ),
            row=2, col=2
        )

        # 5. Feature importance ou Metrics
        if feature_importance is not None and len(feature_importance) > 0:
            top_10 = feature_importance.head(10).sort_values('importance')
            fig.add_trace(
                go.Bar(x=top_10['importance'], y=top_10['feature'], orientation='h', marker_color='#00d4aa'),
                row=3, col=1
            )

        # 6. Rolling Sharpe
        rolling_sharpe = returns.rolling(90).apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0)

        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Rolling Sharpe',
                       line=dict(color='#ffa600', width=2)),
            row=3, col=2
        )

        # Ligne à Sharpe = 1 (threshold)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=3, col=2)

        # Layout global
        fig.update_layout(
            title_text="ML Strategy Dashboard",
            template=self.template,
            height=1200,
            showlegend=True
        )

        return fig

    def save_html(self, fig: go.Figure, path: str):
        """Sauvegarde la figure en HTML"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(str(path))
        logger.info(f"✅ Figure sauvegardée: {path}")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ML VISUALIZER - Test")
    print("=" * 80)

    # Créer données synthétiques
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Equity curve
    daily_returns = np.random.randn(len(dates)) * 0.015 + 0.0005
    equity = pd.Series(100000 * (1 + daily_returns).cumprod(), index=dates)

    # Benchmark
    benchmark_returns = np.random.randn(len(dates)) * 0.012 + 0.0003
    benchmark = pd.Series(100000 * (1 + benchmark_returns).cumprod(), index=dates)

    # Returns
    returns = pd.Series(daily_returns, index=dates)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.rand(20)
    }).sort_values('importance', ascending=False)

    print("\n📊 Création des visualisations...")

    # Créer visualizer
    viz = MLVisualizer(template='plotly_dark')

    # 1. Equity curve
    print("\n1️⃣  Equity curve...")
    fig_equity = viz.plot_equity_curve(equity, benchmark, title="Strategy vs Benchmark")

    # 2. Drawdown
    print("2️⃣  Drawdown...")
    fig_dd = viz.plot_drawdown(equity)

    # 3. Returns distribution
    print("3️⃣  Returns distribution...")
    fig_dist = viz.plot_returns_distribution(returns)

    # 4. Feature importance
    print("4️⃣  Feature importance...")
    fig_fi = viz.plot_feature_importance(feature_importance)

    # 5. Dashboard complet
    print("5️⃣  Dashboard complet...")
    fig_dashboard = viz.create_dashboard(
        equity=equity,
        returns=returns,
        benchmark=benchmark,
        feature_importance=feature_importance
    )

    # Sauvegarder
    # viz.save_html(fig_dashboard, 'ml_models/results/dashboard.html')

    print("\n✅ Visualisations créées!")
    print("   Pour afficher: fig.show()")
    print("   Pour sauvegarder: viz.save_html(fig, 'path.html')")
