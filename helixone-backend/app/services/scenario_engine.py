"""
Scenario Engine - Moteur de simulation de scénarios
Inspiré de BlackRock Aladdin
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time

from app.schemas.scenario import (
    StressTestType,
    PositionImpact,
    RiskMetrics,
    Recommendation,
    ScenarioSimulationResult
)

logger = logging.getLogger(__name__)


class ScenarioEngine:
    """
    Moteur de simulation de scénarios pour stress testing
    """

    # Mapping secteur → sensibilité aux différents chocs
    SECTOR_SENSITIVITIES = {
        "market_crash": {
            "Technology": 1.3,
            "Communication Services": 1.2,
            "Consumer Cyclical": 1.4,
            "Financial Services": 1.5,
            "Real Estate": 1.6,
            "Industrials": 1.2,
            "Basic Materials": 1.3,
            "Energy": 1.5,
            "Consumer Defensive": 0.6,
            "Healthcare": 0.7,
            "Utilities": 0.5
        },
        "rate_shock_positive": {  # Hausse de taux
            "Technology": -1.5,  # Négatif pour tech
            "Real Estate": -1.8,
            "Utilities": -1.3,
            "Financial Services": 0.8,  # Positif pour les banques
            "Consumer Cyclical": -1.0,
            "Consumer Defensive": -0.3,
            "Healthcare": -0.5,
            "Energy": -0.2,
            "Basic Materials": -0.6,
            "Industrials": -0.8,
            "Communication Services": -1.2
        },
        "volatility_spike": {
            "Technology": 1.2,
            "Financial Services": 1.4,
            "Consumer Cyclical": 1.1,
            "Energy": 1.3,
            "Basic Materials": 1.2,
            "Real Estate": 1.1,
            "Consumer Defensive": 0.5,
            "Healthcare": 0.6,
            "Utilities": 0.4,
            "Industrials": 1.0,
            "Communication Services": 1.1
        }
    }

    def __init__(self):
        self.cache = {}  # Cache pour données de marché
        logger.info("ScenarioEngine initialisé")

    def run_stress_test(
        self,
        portfolio: Dict[str, float],
        stress_test_type: StressTestType,
        shock_percent: Optional[float] = None,
        rate_change: Optional[float] = None,
        vix_multiplier: Optional[float] = None
    ) -> ScenarioSimulationResult:
        """
        Exécute un stress test sur le portfolio

        Args:
            portfolio: {ticker: quantity}
            stress_test_type: Type de stress test
            shock_percent: Pour market_crash
            rate_change: Pour rate_shock
            vix_multiplier: Pour volatility_spike

        Returns:
            ScenarioSimulationResult
        """
        start_time = time.time()
        logger.info(f"🚀 Stress test: {stress_test_type.value}")
        logger.info(f"   Portfolio: {len(portfolio)} positions")

        # 1. Collecter les données de marché
        stock_data = self._collect_stock_characteristics(list(portfolio.keys()))

        # 2. Calculer la valeur initiale
        portfolio_value_before = 0
        for ticker, qty in portfolio.items():
            price = stock_data.get(ticker, {}).get('price', 0)
            portfolio_value_before += price * qty

        logger.info(f"   Valeur initiale: ${portfolio_value_before:,.2f}")

        # 3. Simuler selon le type de stress test
        if stress_test_type == StressTestType.MARKET_CRASH:
            position_impacts = self._simulate_market_crash(
                portfolio, stock_data, shock_percent or -30
            )
            scenario_name = f"Crash de Marché {shock_percent or -30}%"

        elif stress_test_type == StressTestType.RATE_SHOCK:
            position_impacts = self._simulate_rate_shock(
                portfolio, stock_data, rate_change or 2.0
            )
            scenario_name = f"Choc de Taux +{rate_change or 2.0}%"

        elif stress_test_type == StressTestType.VOLATILITY_SPIKE:
            position_impacts = self._simulate_volatility_spike(
                portfolio, stock_data, vix_multiplier or 3.0
            )
            scenario_name = f"Spike Volatilité VIX x{vix_multiplier or 3.0}"

        elif stress_test_type == StressTestType.INFLATION_SHOCK:
            position_impacts = self._simulate_inflation_shock(
                portfolio, stock_data
            )
            scenario_name = "Choc d'Inflation"

        else:
            position_impacts = self._simulate_liquidity_crisis(
                portfolio, stock_data
            )
            scenario_name = "Crise de Liquidité"

        # 4. Calculer valeur finale
        portfolio_value_after = sum(pos.value_after for pos in position_impacts)
        impact_amount = portfolio_value_after - portfolio_value_before
        impact_percent = (impact_amount / portfolio_value_before) * 100

        logger.info(f"   Valeur finale: ${portfolio_value_after:,.2f}")
        logger.info(f"   Impact: {impact_percent:+.2f}%")

        # 5. Calculer métriques de risque
        risk_metrics = self._calculate_risk_metrics(
            position_impacts,
            portfolio_value_before,
            portfolio_value_after
        )

        # 6. Générer recommandations
        recommendations = self._generate_recommendations(
            position_impacts,
            stock_data,
            risk_metrics,
            impact_percent
        )

        # 7. Identifier pires/meilleures positions
        worst_position = max(position_impacts, key=lambda p: abs(p.impact_percent))
        best_position = min(position_impacts, key=lambda p: abs(p.impact_percent))

        execution_time_ms = int((time.time() - start_time) * 1000)

        return ScenarioSimulationResult(
            scenario_name=scenario_name,
            scenario_type=stress_test_type.value,
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            impact_percent=impact_percent,
            impact_amount=impact_amount,
            position_impacts=position_impacts,
            risk_metrics=risk_metrics,
            recommendations=recommendations,
            worst_position=f"{worst_position.ticker} ({worst_position.impact_percent:+.1f}%)",
            best_position=f"{best_position.ticker} ({best_position.impact_percent:+.1f}%)",
            execution_time_ms=execution_time_ms
        )

    def _collect_stock_characteristics(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Collecte les caractéristiques des actions (beta, secteur, prix)

        Returns:
            {ticker: {price, beta, sector, market_cap}}
        """
        logger.info(f"📊 Collecte des données pour {len(tickers)} tickers...")
        stock_data = {}

        for ticker in tickers:
            try:
                # Check cache
                if ticker in self.cache:
                    stock_data[ticker] = self.cache[ticker]
                    continue

                stock = yf.Ticker(ticker)
                info = stock.info

                data = {
                    'price': info.get('currentPrice', info.get('regularMarketPrice', 100)),
                    'beta': info.get('beta', 1.0) or 1.0,
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE'),
                    'dividend_yield': info.get('dividendYield', 0) or 0
                }

                stock_data[ticker] = data
                self.cache[ticker] = data  # Cache pour 1 heure

                logger.debug(f"   {ticker}: ${data['price']:.2f}, beta={data['beta']:.2f}, {data['sector']}")

            except Exception as e:
                logger.warning(f"   ⚠️  Erreur {ticker}: {e}")
                # Données par défaut
                stock_data[ticker] = {
                    'price': 100,
                    'beta': 1.0,
                    'sector': 'Unknown',
                    'market_cap': 0,
                    'pe_ratio': None,
                    'dividend_yield': 0
                }

        logger.info(f"   ✅ Données collectées pour {len(stock_data)} positions")
        return stock_data

    def _simulate_market_crash(
        self,
        portfolio: Dict[str, float],
        stock_data: Dict,
        shock_percent: float
    ) -> List[PositionImpact]:
        """
        Simule un crash de marché général

        Impact basé sur:
        - Beta (plus beta est élevé, plus l'impact est fort)
        - Secteur (tech chute plus, defensive moins)
        - Bruit aléatoire pour réalisme
        """
        logger.info(f"💥 Simulation crash de marché: {shock_percent}%")
        position_impacts = []

        for ticker, qty in portfolio.items():
            data = stock_data.get(ticker, {})
            price_before = data.get('price', 100)
            beta = data.get('beta', 1.0)
            sector = data.get('sector', 'Unknown')

            # Impact = base shock × beta × sector multiplier × random noise
            sector_multiplier = self.SECTOR_SENSITIVITIES['market_crash'].get(sector, 1.0)
            random_noise = np.random.normal(1.0, 0.1)  # ±10%

            impact = shock_percent * beta * sector_multiplier * random_noise

            # Appliquer l'impact
            price_after = price_before * (1 + impact / 100)
            price_after = max(price_after, 0.01)  # Pas en négatif

            value_before = price_before * qty
            value_after = price_after * qty
            impact_amount = value_after - value_before

            position_impacts.append(PositionImpact(
                ticker=ticker,
                quantity=qty,
                price_before=price_before,
                price_after=price_after,
                value_before=value_before,
                value_after=value_after,
                impact_percent=impact,
                impact_amount=impact_amount
            ))

            logger.debug(f"   {ticker}: {impact:+.1f}% (beta={beta:.2f}, {sector})")

        return position_impacts

    def _simulate_rate_shock(
        self,
        portfolio: Dict[str, float],
        stock_data: Dict,
        rate_change: float
    ) -> List[PositionImpact]:
        """
        Simule un choc de taux d'intérêt

        Tech/Real Estate: négatif
        Financials: positif
        """
        logger.info(f"📈 Simulation choc de taux: +{rate_change}%")
        position_impacts = []

        for ticker, qty in portfolio.items():
            data = stock_data.get(ticker, {})
            price_before = data.get('price', 100)
            sector = data.get('sector', 'Unknown')

            # Sensibilité sectorielle aux taux
            sector_sensitivity = self.SECTOR_SENSITIVITIES['rate_shock_positive'].get(sector, -0.5)

            # Impact = rate change × sector sensitivity
            impact = rate_change * sector_sensitivity * np.random.normal(1.0, 0.15)

            price_after = price_before * (1 + impact / 100)
            price_after = max(price_after, 0.01)

            value_before = price_before * qty
            value_after = price_after * qty
            impact_amount = value_after - value_before

            position_impacts.append(PositionImpact(
                ticker=ticker,
                quantity=qty,
                price_before=price_before,
                price_after=price_after,
                value_before=value_before,
                value_after=value_after,
                impact_percent=impact,
                impact_amount=impact_amount
            ))

        return position_impacts

    def _simulate_volatility_spike(
        self,
        portfolio: Dict[str, float],
        stock_data: Dict,
        vix_multiplier: float
    ) -> List[PositionImpact]:
        """
        Simule un spike de volatilité (VIX x3, x5, etc.)
        """
        logger.info(f"📊 Simulation spike volatilité: VIX x{vix_multiplier}")
        position_impacts = []

        for ticker, qty in portfolio.items():
            data = stock_data.get(ticker, {})
            price_before = data.get('price', 100)
            beta = data.get('beta', 1.0)
            sector = data.get('sector', 'Unknown')

            # Impact proportionnel au beta et au secteur
            sector_multiplier = self.SECTOR_SENSITIVITIES['volatility_spike'].get(sector, 1.0)
            impact = -10 * beta * sector_multiplier * (vix_multiplier / 3)  # Base: -10% pour VIX x3

            price_after = price_before * (1 + impact / 100)
            price_after = max(price_after, 0.01)

            value_before = price_before * qty
            value_after = price_after * qty
            impact_amount = value_after - value_before

            position_impacts.append(PositionImpact(
                ticker=ticker,
                quantity=qty,
                price_before=price_before,
                price_after=price_after,
                value_before=value_before,
                value_after=value_after,
                impact_percent=impact,
                impact_amount=impact_amount
            ))

        return position_impacts

    def _simulate_inflation_shock(
        self,
        portfolio: Dict[str, float],
        stock_data: Dict
    ) -> List[PositionImpact]:
        """Simule un choc d'inflation"""
        # Inflation favorise: commodities, energy, real estate
        # Défavorise: tech growth, consumer discretionary
        logger.info("📈 Simulation choc d'inflation")
        position_impacts = []

        inflation_impact = {
            "Energy": 15.0,
            "Basic Materials": 10.0,
            "Real Estate": 5.0,
            "Financial Services": 2.0,
            "Utilities": 3.0,
            "Consumer Defensive": -3.0,
            "Healthcare": -5.0,
            "Consumer Cyclical": -12.0,
            "Technology": -18.0,
            "Communication Services": -15.0
        }

        for ticker, qty in portfolio.items():
            data = stock_data.get(ticker, {})
            price_before = data.get('price', 100)
            sector = data.get('sector', 'Unknown')

            impact = inflation_impact.get(sector, 0) * np.random.normal(1.0, 0.2)

            price_after = price_before * (1 + impact / 100)
            price_after = max(price_after, 0.01)

            value_before = price_before * qty
            value_after = price_after * qty
            impact_amount = value_after - value_before

            position_impacts.append(PositionImpact(
                ticker=ticker,
                quantity=qty,
                price_before=price_before,
                price_after=price_after,
                value_before=value_before,
                value_after=value_after,
                impact_percent=impact,
                impact_amount=impact_amount
            ))

        return position_impacts

    def _simulate_liquidity_crisis(
        self,
        portfolio: Dict[str, float],
        stock_data: Dict
    ) -> List[PositionImpact]:
        """Simule une crise de liquidité"""
        logger.info("💧 Simulation crise de liquidité")
        position_impacts = []

        for ticker, qty in portfolio.items():
            data = stock_data.get(ticker, {})
            price_before = data.get('price', 100)
            market_cap = data.get('market_cap', 0)

            # Small caps souffrent plus
            if market_cap < 2e9:  # < $2B
                impact = np.random.uniform(-35, -25)
            elif market_cap < 10e9:  # < $10B
                impact = np.random.uniform(-25, -15)
            else:  # Large cap
                impact = np.random.uniform(-15, -5)

            price_after = price_before * (1 + impact / 100)
            price_after = max(price_after, 0.01)

            value_before = price_before * qty
            value_after = price_after * qty
            impact_amount = value_after - value_before

            position_impacts.append(PositionImpact(
                ticker=ticker,
                quantity=qty,
                price_before=price_before,
                price_after=price_after,
                value_before=value_before,
                value_after=value_after,
                impact_percent=impact,
                impact_amount=impact_amount
            ))

        return position_impacts

    def _calculate_risk_metrics(
        self,
        position_impacts: List[PositionImpact],
        portfolio_value_before: float,
        portfolio_value_after: float
    ) -> RiskMetrics:
        """
        Calcule les métriques de risque
        """
        impacts_pct = [pos.impact_percent for pos in position_impacts]

        var_95 = np.percentile(impacts_pct, 5)
        cvar_95 = np.mean([x for x in impacts_pct if x <= var_95])
        var_99 = np.percentile(impacts_pct, 1)
        cvar_99 = np.mean([x for x in impacts_pct if x <= var_99])

        max_drawdown = min(impacts_pct)

        # Stress Score (0-100, plus c'est élevé, mieux c'est)
        overall_impact = ((portfolio_value_after / portfolio_value_before) - 1) * 100
        if overall_impact >= -10:
            stress_score = 90
        elif overall_impact >= -20:
            stress_score = 75
        elif overall_impact >= -30:
            stress_score = 60
        elif overall_impact >= -40:
            stress_score = 40
        else:
            stress_score = 20

        # Recovery time estimé (jours)
        if overall_impact >= -20:
            recovery_days = 90
        elif overall_impact >= -30:
            recovery_days = 180
        elif overall_impact >= -50:
            recovery_days = 365
        else:
            recovery_days = 730

        return RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            stress_score=stress_score,
            recovery_time_days=recovery_days
        )

    def _generate_recommendations(
        self,
        position_impacts: List[PositionImpact],
        stock_data: Dict,
        risk_metrics: RiskMetrics,
        overall_impact: float
    ) -> List[Recommendation]:
        """
        Génère des recommandations automatiques
        """
        recommendations = []

        # 1. Hedging si impact sévère
        if overall_impact < -25:
            recommendations.append(Recommendation(
                type="hedging",
                priority="high",
                title="Protéger le portfolio avec un hedge",
                description=f"Impact estimé de {overall_impact:.1f}%. Considérer l'achat de positions inverses (SQQQ, SPXS) ou d'options put.",
                impact_reduction=abs(overall_impact) * 0.3
            ))

        # 2. Diversification si concentration sectorielle
        sectors = [stock_data.get(pos.ticker, {}).get('sector', 'Unknown') for pos in position_impacts]
        sector_counts = pd.Series(sectors).value_counts()
        if len(sector_counts) > 0:
            most_common_sector = sector_counts.index[0]
            sector_pct = (sector_counts.iloc[0] / len(sectors)) * 100

            if sector_pct > 50:
                recommendations.append(Recommendation(
                    type="diversification",
                    priority="high",
                    title=f"Réduire la concentration en {most_common_sector}",
                    description=f"{sector_pct:.0f}% du portfolio dans un seul secteur. Diversifier vers d'autres secteurs (Utilities, Healthcare, Consumer Defensive).",
                    impact_reduction=15.0
                ))

        # 3. Réduire positions haut beta
        high_beta_positions = [pos for pos in position_impacts if stock_data.get(pos.ticker, {}).get('beta', 1) > 1.5]
        if len(high_beta_positions) > 0 and len(high_beta_positions) / len(position_impacts) > 0.3:
            recommendations.append(Recommendation(
                type="position_sizing",
                priority="medium",
                title="Réduire l'exposition aux positions à haut beta",
                description=f"{len(high_beta_positions)} positions avec beta > 1.5. Ces actions amplifient les mouvements du marché.",
                impact_reduction=10.0
            ))

        # 4. Ajouter défensives si stress score faible
        if risk_metrics.stress_score < 60:
            recommendations.append(Recommendation(
                type="diversification",
                priority="medium",
                title="Ajouter des actifs défensifs",
                description="Score de stress faible. Considérer l'ajout d'actions défensives: JNJ, PG, KO, WMT, ou ETFs de secteurs défensifs.",
                impact_reduction=12.0
            ))

        return recommendations
