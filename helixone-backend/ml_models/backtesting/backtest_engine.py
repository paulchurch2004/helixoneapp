"""
Backtest Engine - Backtrader pour tester les stratégies ML

Features:
- Backtesting réaliste avec slippage et commissions
- Stratégie ML basée sur les signaux d'ensemble
- Métriques: Sharpe ratio, max drawdown, win rate, etc.
- Visualisation des résultats

Utilisation:
    engine = BacktestEngine()
    results = engine.run_backtest(
        ticker='AAPL',
        model_path='ml_models/saved_models/AAPL/ensemble',
        initial_cash=100000
    )
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
from datetime import datetime
import sys

# Imports relatifs
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ensemble_model import MultiHorizonEnsemble
from data_collection.data_cache import DataCache
from feature_engineering.technical_indicators import TechnicalIndicators
from feature_engineering.macro_features import MacroFeatures
from feature_engineering.sentiment_features import SentimentFeatures
from feature_engineering.volume_features import VolumeFeatures

logger = logging.getLogger(__name__)


class MLStrategy(bt.Strategy):
    """
    Stratégie Backtrader basée sur les signaux ML
    """

    params = (
        ('model_path', None),
        ('features', []),
        ('confidence_threshold', 0.6),  # Confiance minimum pour trader
        ('position_size', 0.95),  # Utiliser 95% du capital disponible
        ('stop_loss_pct', -0.10),  # Stop loss à -10%
        ('take_profit_pct', 0.20),  # Take profit à +20%
    )

    def __init__(self):
        """Initialise la stratégie"""
        self.model = None
        self.data_df = None
        self.order = None
        self.buy_price = None
        self.buy_comm = None

        # Charger le modèle
        if self.params.model_path:
            try:
                self.model = MultiHorizonEnsemble()
                self.model.load_all(self.params.model_path)
                logger.info(f"✅ Modèle chargé: {self.params.model_path}")
            except Exception as e:
                logger.error(f"❌ Erreur chargement modèle: {e}")
                raise

        # Métriques
        self.trades = []
        self.signals = []

    def log(self, txt, dt=None):
        """Log avec timestamp"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.debug(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        """Callback quand un ordre est exécuté"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                self.log(f'BUY EXECUTED: ${order.executed.price:.2f}, Comm: ${order.executed.comm:.2f}')

            else:  # Sell
                profit = order.executed.price - self.buy_price
                profit_pct = (profit / self.buy_price) * 100
                self.log(f'SELL EXECUTED: ${order.executed.price:.2f}, Profit: ${profit:.2f} ({profit_pct:+.2f}%)')

                # Enregistrer trade
                self.trades.append({
                    'buy_price': self.buy_price,
                    'sell_price': order.executed.price,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'comm': self.buy_comm + order.executed.comm
                })

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'ORDER CANCELED/MARGIN/REJECTED')

        self.order = None

    def next(self):
        """Logique de trading à chaque période"""
        # Si ordre en cours, attendre
        if self.order:
            return

        # Pas de modèle = pas de trading
        if self.model is None:
            return

        # Obtenir données historiques
        current_idx = len(self.datas[0])

        # Besoin d'au moins 60 jours d'historique pour LSTM (lookback=30)
        if current_idx < 60:
            return

        # Construire DataFrame avec historique récent (60 jours)
        df_recent = self._get_recent_dataframe(lookback=60)

        if df_recent is None or len(df_recent) < 60:
            return

        # Générer signal ML
        try:
            signal = self.model.get_multi_horizon_signals(df_recent, self.params.features)
        except Exception as e:
            logger.warning(f"Erreur génération signal: {e}")
            return

        # Extraire consensus
        consensus = signal['consensus']
        action = consensus['action']
        confidence = consensus['confidence']

        # Enregistrer signal
        self.signals.append({
            'date': self.datas[0].datetime.date(0),
            'action': action,
            'confidence': confidence,
            'direction': consensus['direction']
        })

        # Trading logic
        if not self.position:
            # Pas de position: chercher signal BUY

            if action == 'BUY' and confidence >= self.params.confidence_threshold:
                # Calculer taille position
                cash = self.broker.getcash()
                size = int((cash * self.params.position_size) / self.datas[0].close[0])

                if size > 0:
                    self.log(f'BUY SIGNAL: {action} (confidence: {confidence:.2%})')
                    self.order = self.buy(size=size)

        else:
            # Position ouverte: chercher signal SELL ou stop loss/take profit

            current_price = self.datas[0].close[0]
            price_change_pct = (current_price - self.buy_price) / self.buy_price

            # Stop loss
            if price_change_pct <= self.params.stop_loss_pct:
                self.log(f'STOP LOSS: {price_change_pct:.2%}')
                self.order = self.sell(size=self.position.size)

            # Take profit
            elif price_change_pct >= self.params.take_profit_pct:
                self.log(f'TAKE PROFIT: {price_change_pct:.2%}')
                self.order = self.sell(size=self.position.size)

            # Signal SELL du modèle
            elif action == 'SELL' and confidence >= self.params.confidence_threshold:
                self.log(f'SELL SIGNAL: {action} (confidence: {confidence:.2%})')
                self.order = self.sell(size=self.position.size)

    def _get_recent_dataframe(self, lookback: int = 60) -> Optional[pd.DataFrame]:
        """
        Construit un DataFrame avec l'historique récent

        Args:
            lookback: Nombre de jours à retourner

        Returns:
            DataFrame avec features
        """
        if self.data_df is None:
            return None

        # Index actuel
        current_date = self.datas[0].datetime.date(0)

        # Trouver l'index dans data_df
        try:
            df_idx = self.data_df.index.get_loc(pd.Timestamp(current_date))
        except KeyError:
            return None

        # Extraire derniers lookback jours
        start_idx = max(0, df_idx - lookback + 1)
        end_idx = df_idx + 1

        df_recent = self.data_df.iloc[start_idx:end_idx]

        return df_recent


class BacktestEngine:
    """
    Moteur de backtesting pour stratégies ML
    """

    def __init__(self):
        """Initialise le moteur"""
        self.data_cache = DataCache()
        self.tech_indicators = TechnicalIndicators()
        self.macro_features = MacroFeatures()
        self.sentiment_features = SentimentFeatures()
        self.volume_features = VolumeFeatures()

        logger.info("BacktestEngine initialisé")

    def prepare_data(
        self,
        ticker: str,
        start_date: str = '2018-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prépare les données pour le backtest

        Args:
            ticker: Ticker
            start_date: Date de début
            end_date: Date de fin (None = aujourd'hui)

        Returns:
            DataFrame avec toutes les features
        """
        logger.info(f"Préparation données: {ticker} ({start_date} → {end_date or 'now'})")

        # Télécharger données
        dataset = self.data_cache.get_ml_dataset(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            include_macro=True
        )

        df = dataset[ticker]

        # Ajouter features
        df = self.tech_indicators.add_all_indicators(df)
        df = self.volume_features.add_volume_features(df)
        df = self.sentiment_features.add_sentiment_features(df, ticker)

        # Nettoyer NaN
        df = df.dropna()

        logger.info(f"   ✅ {len(df)} jours, {len(df.columns)} features")

        return df

    def run_backtest(
        self,
        ticker: str,
        model_path: str,
        features: List[str],
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        initial_cash: float = 100000,
        commission: float = 0.001,  # 0.1%
        confidence_threshold: float = 0.6,
        stop_loss_pct: float = -0.10,
        take_profit_pct: float = 0.20
    ) -> Dict:
        """
        Exécute le backtest

        Args:
            ticker: Ticker à tester
            model_path: Chemin du modèle ensemble
            features: Liste des features
            start_date: Date de début backtest
            end_date: Date de fin (None = aujourd'hui)
            initial_cash: Capital initial
            commission: Commission par trade (0.001 = 0.1%)
            confidence_threshold: Confiance minimum
            stop_loss_pct: Stop loss
            take_profit_pct: Take profit

        Returns:
            Dict avec résultats
        """
        logger.info("=" * 80)
        logger.info(f"BACKTEST: {ticker}")
        logger.info("=" * 80)
        logger.info(f"   Période: {start_date} → {end_date or 'now'}")
        logger.info(f"   Capital initial: ${initial_cash:,.0f}")
        logger.info(f"   Commission: {commission*100:.2f}%")
        logger.info(f"   Modèle: {model_path}")

        # Préparer données
        df = self.prepare_data(ticker, start_date, end_date)

        # Créer Cerebro
        cerebro = bt.Cerebro()

        # Ajouter données à Backtrader
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)

        # Ajouter stratégie
        cerebro.addstrategy(
            MLStrategy,
            model_path=model_path,
            features=features,
            confidence_threshold=confidence_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

        # Passer le DataFrame à la stratégie (via strat.data_df)
        # Note: On devra le faire dans run() ci-dessous

        # Configuration broker
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)

        # Ajouter analyseurs
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Valeur initiale
        start_value = cerebro.broker.getvalue()

        logger.info(f"\n📊 Valeur initiale: ${start_value:,.2f}")
        logger.info("   Exécution backtest...\n")

        # Run!
        strats = cerebro.run()
        strat = strats[0]

        # Injecter DataFrame dans stratégie (hack)
        strat.data_df = df

        # Valeur finale
        end_value = cerebro.broker.getvalue()

        # Extraire métriques
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades_analyzer = strat.analyzers.trades.get_analysis()

        # Calculer métriques
        total_return = ((end_value - start_value) / start_value) * 100

        total_trades = trades_analyzer.get('total', {}).get('total', 0)
        won_trades = trades_analyzer.get('won', {}).get('total', 0)
        lost_trades = trades_analyzer.get('lost', {}).get('total', 0)

        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        # Sharpe ratio
        sharpe_ratio = sharpe.get('sharperatio', None)

        # Max drawdown
        max_dd = drawdown.get('max', {}).get('drawdown', 0)

        # Résultats
        results = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': initial_cash,
            'final_value': end_value,
            'total_return': total_return,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'won_trades': won_trades,
            'lost_trades': lost_trades,
            'win_rate': win_rate,
            'trades': strat.trades,
            'signals': strat.signals
        }

        # Afficher résultats
        logger.info("=" * 80)
        logger.info("📊 RÉSULTATS BACKTEST")
        logger.info("=" * 80)
        logger.info(f"   Valeur finale: ${end_value:,.2f}")
        logger.info(f"   Return total: {total_return:+.2f}%")
        logger.info(f"   Sharpe ratio: {sharpe_ratio:.2f}" if sharpe_ratio else "   Sharpe ratio: N/A")
        logger.info(f"   Max drawdown: {max_dd:.2f}%")
        logger.info(f"   Total trades: {total_trades}")
        logger.info(f"   Win rate: {win_rate:.1f}%")

        return results


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("BACKTEST ENGINE - Test")
    print("=" * 80)

    # Note: Cet exemple nécessite un modèle déjà entraîné
    # Utiliser model_trainer.py d'abord

    engine = BacktestEngine()

    # Example (nécessite modèle pré-entraîné)
    # results = engine.run_backtest(
    #     ticker='AAPL',
    #     model_path='ml_models/saved_models/AAPL/ensemble',
    #     features=['rsi_14', 'macd', 'bb_width', 'sma_20'],  # Exemple
    #     start_date='2022-01-01',
    #     initial_cash=100000
    # )

    print("\n✅ BacktestEngine prêt!")
    print("   Pour utiliser:")
    print("   1. Entraîner un modèle avec model_trainer.py")
    print("   2. Lancer backtest avec run_backtest()")
