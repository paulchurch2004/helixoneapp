"""
Service de collecte de données via Alpha Vantage API
Fournit des données de marché de haute qualité (500 requêtes/jour gratuit)
"""

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import os
import pandas as pd
import time

logger = logging.getLogger(__name__)

# Clé API Alpha Vantage (gratuit: https://www.alphavantage.co/support/#api-key)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")  # Remplacer par votre clé

class AlphaVantageCollector:
    """
    Collecteur de données Alpha Vantage

    Features:
    - Prix OHLCV journaliers et intraday
    - Données fondamentales (états financiers, ratios)
    - Indicateurs techniques pré-calculés
    - Forex et crypto

    Limites gratuites:
    - 500 requêtes/jour
    - 5 requêtes/minute
    """

    def __init__(self, api_key: str = ALPHA_VANTAGE_API_KEY):
        """
        Initialiser le collecteur Alpha Vantage

        Args:
            api_key: Clé API Alpha Vantage
        """
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')
        self.ti = TechIndicators(key=api_key, output_format='pandas')

        # Rate limiting: 5 requêtes/minute max
        self.min_request_interval = 12  # secondes
        self.last_request_time = 0

        # Compteurs
        self.requests_today = 0
        self.max_requests_per_day = 500

        logger.info(f"✅ AlphaVantageCollector initialisé (clé: {api_key[:8]}...)")

    def _rate_limit(self):
        """Respecter les limites de taux (5 req/min)"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            logger.debug(f"⏳ Rate limiting: attente {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
        self.requests_today += 1

        if self.requests_today >= self.max_requests_per_day:
            logger.warning(f"⚠️ Limite quotidienne atteinte ({self.max_requests_per_day} req/jour)")

    def get_daily_data(
        self,
        symbol: str,
        outputsize: str = 'full'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Récupérer les données journalières

        Args:
            symbol: Symbole du ticker (ex: "AAPL")
            outputsize: 'compact' (100 jours) ou 'full' (20+ ans)

        Returns:
            (dataframe, metadata)
        """
        try:
            self._rate_limit()

            logger.info(f"📊 Alpha Vantage: Collecte daily data pour {symbol}")
            data, meta_data = self.ts.get_daily_adjusted(
                symbol=symbol,
                outputsize=outputsize
            )

            # Renommer les colonnes pour correspondre à notre schéma
            data.columns = [
                'open', 'high', 'low', 'close',
                'adjusted_close', 'volume', 'dividend', 'split'
            ]

            # Réinitialiser l'index pour avoir la date comme colonne
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'timestamp'}, inplace=True)

            logger.info(f"✅ {symbol}: {len(data)} jours collectés")

            return data, meta_data

        except Exception as e:
            logger.error(f"❌ Erreur collecte Alpha Vantage {symbol}: {e}")
            raise

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = '5min',
        outputsize: str = 'full'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Récupérer les données intraday

        Args:
            symbol: Symbole du ticker
            interval: '1min', '5min', '15min', '30min', '60min'
            outputsize: 'compact' (100 points) ou 'full' (jours complets)

        Returns:
            (dataframe, metadata)
        """
        try:
            self._rate_limit()

            logger.info(f"📊 Alpha Vantage: Collecte intraday {interval} pour {symbol}")
            data, meta_data = self.ts.get_intraday(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize
            )

            # Renommer les colonnes
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'timestamp'}, inplace=True)

            logger.info(f"✅ {symbol}: {len(data)} points intraday collectés")

            return data, meta_data

        except Exception as e:
            logger.error(f"❌ Erreur collecte intraday Alpha Vantage {symbol}: {e}")
            raise

    def get_daily(self, symbol: str, outputsize: str = 'full') -> Dict:
        """
        Alias pour get_daily_data() qui retourne au format dict

        Args:
            symbol: Symbole du ticker
            outputsize: 'compact' ou 'full'

        Returns:
            Dict avec Time Series (Daily)
        """
        try:
            self._rate_limit()

            logger.info(f"📊 Alpha Vantage: Daily data pour {symbol}")
            data, meta_data = self.ts.get_daily_adjusted(
                symbol=symbol,
                outputsize=outputsize
            )

            # Convertir en format dict compatible
            result = {
                'Meta Data': meta_data,
                'Time Series (Daily)': {}
            }

            # Convertir dataframe en dict
            for date, row in data.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                result['Time Series (Daily)'][date_str] = {
                    '1. open': str(row.get('1. open', '')),
                    '2. high': str(row.get('2. high', '')),
                    '3. low': str(row.get('3. low', '')),
                    '4. close': str(row.get('4. close', '')),
                    '5. adjusted close': str(row.get('5. adjusted close', '')),
                    '6. volume': str(row.get('6. volume', '')),
                    '7. dividend amount': str(row.get('7. dividend amount', '')),
                    '8. split coefficient': str(row.get('8. split coefficient', ''))
                }

            logger.info(f"✅ {symbol}: {len(result['Time Series (Daily)'])} jours")

            return result

        except Exception as e:
            logger.error(f"❌ Erreur daily Alpha Vantage {symbol}: {e}")
            raise

    def get_quote(self, symbol: str) -> Dict:
        """
        Récupérer la quote en temps réel

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec prix, volume, timestamp
        """
        try:
            self._rate_limit()

            logger.info(f"💹 Alpha Vantage: Quote temps réel pour {symbol}")
            data, _ = self.ts.get_quote_endpoint(symbol=symbol)

            quote = {
                'symbol': data['01. symbol'][0],
                'price': float(data['05. price'][0]),
                'volume': int(data['06. volume'][0]),
                'timestamp': data['07. latest trading day'][0],
                'change': float(data['09. change'][0]),
                'change_percent': data['10. change percent'][0]
            }

            logger.info(f"✅ {symbol}: ${quote['price']:.2f}")

            return quote

        except Exception as e:
            logger.error(f"❌ Erreur quote Alpha Vantage {symbol}: {e}")
            raise

    def get_company_overview(self, symbol: str) -> Dict:
        """
        Récupérer les informations fondamentales de l'entreprise

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec données fondamentales (secteur, industrie, market cap, etc.)
        """
        try:
            self._rate_limit()

            logger.info(f"🏢 Alpha Vantage: Company overview pour {symbol}")
            data, _ = self.fd.get_company_overview(symbol=symbol)

            overview = {
                'symbol': data['Symbol'][0],
                'name': data['Name'][0],
                'description': data['Description'][0],
                'sector': data['Sector'][0],
                'industry': data['Industry'][0],
                'market_cap': int(data['MarketCapitalization'][0]) if data['MarketCapitalization'][0] != 'None' else 0,
                'pe_ratio': float(data['PERatio'][0]) if data['PERatio'][0] != 'None' else None,
                'peg_ratio': float(data['PEGRatio'][0]) if data['PEGRatio'][0] != 'None' else None,
                'book_value': float(data['BookValue'][0]) if data['BookValue'][0] != 'None' else None,
                'dividend_yield': float(data['DividendYield'][0]) if data['DividendYield'][0] != 'None' else None,
                'eps': float(data['EPS'][0]) if data['EPS'][0] != 'None' else None,
                'revenue_per_share': float(data['RevenuePerShareTTM'][0]) if data['RevenuePerShareTTM'][0] != 'None' else None,
                'profit_margin': float(data['ProfitMargin'][0]) if data['ProfitMargin'][0] != 'None' else None,
                'beta': float(data['Beta'][0]) if data['Beta'][0] != 'None' else None,
                '52_week_high': float(data['52WeekHigh'][0]) if data['52WeekHigh'][0] != 'None' else None,
                '52_week_low': float(data['52WeekLow'][0]) if data['52WeekLow'][0] != 'None' else None,
                '50_day_ma': float(data['50DayMovingAverage'][0]) if data['50DayMovingAverage'][0] != 'None' else None,
                '200_day_ma': float(data['200DayMovingAverage'][0]) if data['200DayMovingAverage'][0] != 'None' else None,
            }

            logger.info(f"✅ {symbol}: {overview['name']} ({overview['sector']})")

            return overview

        except Exception as e:
            logger.error(f"❌ Erreur company overview Alpha Vantage {symbol}: {e}")
            raise

    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """
        Récupérer le compte de résultat (Income Statement)

        Args:
            symbol: Symbole du ticker

        Returns:
            DataFrame avec les données annuelles et trimestrielles
        """
        try:
            self._rate_limit()

            logger.info(f"💰 Alpha Vantage: Income statement pour {symbol}")
            data, _ = self.fd.get_income_statement_annual(symbol=symbol)

            logger.info(f"✅ {symbol}: {len(data)} années de données")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur income statement Alpha Vantage {symbol}: {e}")
            raise

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """
        Récupérer le bilan (Balance Sheet)

        Args:
            symbol: Symbole du ticker

        Returns:
            DataFrame avec le bilan annuel
        """
        try:
            self._rate_limit()

            logger.info(f"📋 Alpha Vantage: Balance sheet pour {symbol}")
            data, _ = self.fd.get_balance_sheet_annual(symbol=symbol)

            logger.info(f"✅ {symbol}: {len(data)} années de bilan")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur balance sheet Alpha Vantage {symbol}: {e}")
            raise

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """
        Récupérer les flux de trésorerie (Cash Flow)

        Args:
            symbol: Symbole du ticker

        Returns:
            DataFrame avec les flux de trésorerie annuels
        """
        try:
            self._rate_limit()

            logger.info(f"💵 Alpha Vantage: Cash flow pour {symbol}")
            data, _ = self.fd.get_cash_flow_annual(symbol=symbol)

            logger.info(f"✅ {symbol}: {len(data)} années de cash flow")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur cash flow Alpha Vantage {symbol}: {e}")
            raise

    def get_rsi(self, symbol: str, interval: str = 'daily', time_period: int = 14) -> pd.DataFrame:
        """
        Récupérer le RSI (Relative Strength Index)

        Args:
            symbol: Symbole du ticker
            interval: 'daily', '60min', '30min', '15min', '5min', '1min'
            time_period: Période de calcul (défaut: 14)

        Returns:
            DataFrame avec le RSI
        """
        try:
            self._rate_limit()

            logger.info(f"📈 Alpha Vantage: RSI pour {symbol}")
            data, _ = self.ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period)

            logger.info(f"✅ {symbol}: {len(data)} points RSI")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur RSI Alpha Vantage {symbol}: {e}")
            raise

    def get_macd(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """
        Récupérer le MACD (Moving Average Convergence Divergence)

        Args:
            symbol: Symbole du ticker
            interval: 'daily', '60min', '30min', '15min', '5min', '1min'

        Returns:
            DataFrame avec MACD, signal, histogram
        """
        try:
            self._rate_limit()

            logger.info(f"📊 Alpha Vantage: MACD pour {symbol}")
            data, _ = self.ti.get_macd(symbol=symbol, interval=interval)

            logger.info(f"✅ {symbol}: {len(data)} points MACD")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur MACD Alpha Vantage {symbol}: {e}")
            raise

    def get_bbands(self, symbol: str, interval: str = 'daily', time_period: int = 20) -> pd.DataFrame:
        """
        Récupérer les Bollinger Bands

        Args:
            symbol: Symbole du ticker
            interval: 'daily', '60min', '30min', '15min', '5min', '1min'
            time_period: Période de calcul (défaut: 20)

        Returns:
            DataFrame avec upper, middle, lower bands
        """
        try:
            self._rate_limit()

            logger.info(f"📉 Alpha Vantage: Bollinger Bands pour {symbol}")
            data, _ = self.ti.get_bbands(symbol=symbol, interval=interval, time_period=time_period)

            logger.info(f"✅ {symbol}: {len(data)} points Bollinger Bands")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur Bollinger Bands Alpha Vantage {symbol}: {e}")
            raise

    def get_usage_stats(self) -> Dict:
        """
        Obtenir les statistiques d'utilisation

        Returns:
            Dict avec nombre de requêtes et limite
        """
        return {
            'requests_today': self.requests_today,
            'max_requests_per_day': self.max_requests_per_day,
            'remaining': self.max_requests_per_day - self.requests_today,
            'percentage_used': (self.requests_today / self.max_requests_per_day) * 100
        }

    # === COMMODITIES DATA ===

    def _get_commodity_data(self, function: str, interval: str = 'monthly') -> pd.DataFrame:
        """
        Generic method to fetch commodity data

        Args:
            function: Commodity function name (e.g., 'WTI', 'BRENT', 'NATURAL_GAS')
            interval: 'daily', 'weekly', 'monthly'

        Returns:
            DataFrame with commodity prices
        """
        try:
            self._rate_limit()

            import requests

            logger.info(f"📊 Alpha Vantage: Commodity {function} data ({interval})")

            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function,
                'interval': interval,
                'apikey': self.api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data_dict = response.json()

            # Check for errors
            if 'Error Message' in data_dict:
                raise Exception(f"API Error: {data_dict['Error Message']}")

            if 'Note' in data_dict:
                raise Exception(f"API Rate Limit: {data_dict['Note']}")

            # Extract data key (usually 'data')
            data_key = 'data'
            if data_key not in data_dict:
                raise Exception(f"Unexpected response format: {list(data_dict.keys())}")

            # Convert to DataFrame
            df = pd.DataFrame(data_dict[data_key])

            logger.info(f"✅ {function}: {len(df)} points collectés")

            return df

        except Exception as e:
            logger.error(f"❌ Erreur commodity {function}: {e}")
            raise

    def get_wti_crude_oil(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get WTI Crude Oil prices

        Args:
            interval: 'daily', 'weekly', 'monthly'

        Returns:
            DataFrame with date, value (USD per barrel)
        """
        return self._get_commodity_data('WTI', interval)

    def get_brent_crude_oil(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Brent Crude Oil prices

        Args:
            interval: 'daily', 'weekly', 'monthly'

        Returns:
            DataFrame with date, value (USD per barrel)
        """
        return self._get_commodity_data('BRENT', interval)

    def get_natural_gas(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Natural Gas prices (Henry Hub)

        Args:
            interval: 'daily', 'weekly', 'monthly'

        Returns:
            DataFrame with date, value (USD per million BTU)
        """
        return self._get_commodity_data('NATURAL_GAS', interval)

    def get_copper(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Copper prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per pound)
        """
        return self._get_commodity_data('COPPER', interval)

    def get_aluminum(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Aluminum prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per metric ton)
        """
        return self._get_commodity_data('ALUMINUM', interval)

    def get_wheat(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Wheat prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per bushel)
        """
        return self._get_commodity_data('WHEAT', interval)

    def get_corn(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Corn prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per bushel)
        """
        return self._get_commodity_data('CORN', interval)

    def get_cotton(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Cotton prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per pound)
        """
        return self._get_commodity_data('COTTON', interval)

    def get_sugar(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Sugar prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per pound)
        """
        return self._get_commodity_data('SUGAR', interval)

    def get_coffee(self, interval: str = 'monthly') -> pd.DataFrame:
        """
        Get Coffee prices

        Args:
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with date, value (USD per pound)
        """
        return self._get_commodity_data('COFFEE', interval)

    def get_all_commodities(self, interval: str = 'monthly') -> Dict[str, pd.DataFrame]:
        """
        Get prices for all major commodities

        Args:
            interval: 'monthly' recommended (to limit API calls)

        Returns:
            Dictionary with commodity name as key and DataFrame as value

        Note:
            This will consume 10 API calls. Use sparingly.
        """
        commodities = {
            'wti_crude_oil': self.get_wti_crude_oil(interval),
            'brent_crude_oil': self.get_brent_crude_oil(interval),
            'natural_gas': self.get_natural_gas(interval),
            'copper': self.get_copper(interval),
            'aluminum': self.get_aluminum(interval),
            'wheat': self.get_wheat(interval),
            'corn': self.get_corn(interval),
            'cotton': self.get_cotton(interval),
            'sugar': self.get_sugar(interval),
            'coffee': self.get_coffee(interval)
        }

        return commodities


# Instance globale pour réutilisation
_alpha_vantage_collector = None

def get_alpha_vantage_collector(api_key: str = None) -> AlphaVantageCollector:
    """
    Obtenir l'instance du collecteur Alpha Vantage (singleton)

    Args:
        api_key: Clé API (optionnel, utilise variable d'environnement par défaut)

    Returns:
        Instance AlphaVantageCollector
    """
    global _alpha_vantage_collector

    if _alpha_vantage_collector is None:
        _alpha_vantage_collector = AlphaVantageCollector(api_key=api_key or ALPHA_VANTAGE_API_KEY)

    return _alpha_vantage_collector
