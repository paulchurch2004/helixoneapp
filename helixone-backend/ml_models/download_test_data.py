#!/usr/bin/env python3
"""
Télécharge des données de test pour entraînement ML

Utilise yfinance avec retry et fallback vers données synthétiques
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_with_retry(ticker, start_date, end_date, max_retries=3):
    """Télécharge avec retry"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Tentative {attempt + 1}/{max_retries} pour {ticker}...")

            # Télécharger avec période plus large
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False
            )

            if not data.empty:
                logger.info(f"✅ {len(data)} jours téléchargés pour {ticker}")
                return data
            else:
                logger.warning(f"Pas de données pour {ticker}, tentative {attempt + 1}")

        except Exception as e:
            logger.error(f"Erreur téléchargement {ticker}: {e}")

    return None


def generate_synthetic_data(ticker, start_date, end_date):
    """Génère des données synthétiques réalistes"""
    logger.info(f"📊 Génération de données synthétiques pour {ticker}...")

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Filtre jours de semaine seulement
    dates = dates[dates.weekday < 5]

    np.random.seed(hash(ticker) % 2**32)

    # Prix de base selon le ticker
    base_prices = {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 130.0,
        'AMZN': 140.0,
        'TSLA': 200.0
    }

    initial_price = base_prices.get(ticker, 100.0)

    # Générer returns avec drift positif et volatilité
    daily_returns = np.random.randn(len(dates)) * 0.02 + 0.0003  # ~7% CAGR, 20% vol

    # Prix (cumulative)
    prices = initial_price * (1 + daily_returns).cumprod()

    # Volume
    avg_volume = 50_000_000
    volumes = avg_volume * (1 + np.random.randn(len(dates)) * 0.3)
    volumes = np.maximum(volumes, 1_000_000)  # Minimum 1M

    # OHLC
    high_offset = np.random.uniform(0.01, 0.03, len(dates))
    low_offset = np.random.uniform(-0.03, -0.01, len(dates))
    open_offset = np.random.uniform(-0.01, 0.01, len(dates))

    df = pd.DataFrame({
        'Open': prices * (1 + open_offset),
        'High': prices * (1 + high_offset),
        'Low': prices * (1 + low_offset),
        'Close': prices,
        'Adj Close': prices,
        'Volume': volumes.astype(int)
    }, index=dates)

    logger.info(f"✅ {len(df)} jours de données synthétiques générées")
    logger.info(f"   Prix: ${df['Close'].iloc[0]:.2f} → ${df['Close'].iloc[-1]:.2f}")
    logger.info(f"   Return total: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:+.1f}%")

    return df


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    print("=" * 80)
    print("📥 TÉLÉCHARGEMENT DONNÉES DE TEST")
    print("=" * 80)

    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Ticker: {ticker}")
        print("=" * 80)

        # Essayer téléchargement réel
        data = download_with_retry(ticker, start_date, end_date)

        # Fallback vers synthétique
        if data is None or data.empty:
            logger.warning(f"⚠️ Yahoo Finance échoué, utilisation données synthétiques")
            data = generate_synthetic_data(ticker, start_date, end_date)

        # Sauvegarder
        output_file = f'data/{ticker}_historical.csv'
        data.to_csv(output_file)
        logger.info(f"💾 Sauvegardé: {output_file}")

        print(f"\n📊 Résumé {ticker}:")
        print(f"   Période: {data.index[0]} → {data.index[-1]}")
        print(f"   Jours: {len(data)}")
        print(f"   Prix moyen: ${data['Close'].mean():.2f}")
        print(f"   Volume moyen: {data['Volume'].mean()/1e6:.1f}M")

    print("\n" + "=" * 80)
    print("✅ TERMINÉ")
    print("=" * 80)
