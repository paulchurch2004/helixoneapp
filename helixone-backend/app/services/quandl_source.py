"""
Quandl (Nasdaq Data Link) Data Source
Documentation: https://docs.data.nasdaq.com/

Free Tier Limits:
- 50 API calls per day (with free API key)
- 20 API calls per day (anonymous/no key)
- Rate limit: 300 calls per 10 seconds, 2,000 per 10 minutes

Free Datasets:
- Commodity prices (gold, oil, silver, natural gas)
- Economic indicators
- World Bank commodity data
- CME futures (limited free access)

API Key: Optional but recommended (free registration)
Coverage: 400+ free datasets, 50,000+ premium datasets
"""

import os
import time
import requests
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta


class QuandlSource:
    """
    Quandl/Nasdaq Data Link collector for commodities and economic data

    Free Tier: 50 calls/day with API key, 20/day without
    Coverage: Gold, oil, silver, natural gas, economic indicators
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Quandl source

        Args:
            api_key: Quandl/Nasdaq Data Link API key (optional, recommended for 50/day limit)
        """
        self.api_key = api_key or os.getenv('QUANDL_API_KEY')
        self.base_url = "https://data.nasdaq.com/api/v3"

        # Rate limiting: 50 calls/day = ~2 calls/hour conservatively
        # We'll use 30 second intervals to be safe
        self.min_request_interval = 30.0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with rate limiting

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Quandl request failed: {str(e)}")

    def get_dataset(
        self,
        database_code: str,
        dataset_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        collapse: Optional[str] = None,
        limit: Optional[int] = None,
        order: str = 'asc'
    ) -> Dict:
        """
        Get time series dataset

        Args:
            database_code: Database code (e.g., 'LBMA', 'CHRIS', 'ODA')
            dataset_code: Dataset code (e.g., 'GOLD', 'CME_CL1')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            collapse: Data frequency - 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
            limit: Number of rows to return
            order: 'asc' or 'desc'

        Returns:
            Dictionary with dataset metadata and data

        Example:
            >>> # Get gold prices
            >>> gold = quandl.get_dataset('LBMA', 'GOLD', limit=30)
        """
        endpoint = f"datasets/{database_code}/{dataset_code}/data.json"

        params = {
            'order': order
        }

        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if collapse:
            params['collapse'] = collapse
        if limit:
            params['limit'] = limit

        result = self._make_request(endpoint, params)

        return result.get('dataset_data', {})

    def get_dataset_metadata(self, database_code: str, dataset_code: str) -> Dict:
        """
        Get dataset metadata (column names, description, etc.)

        Args:
            database_code: Database code
            dataset_code: Dataset code

        Returns:
            Dictionary with metadata
        """
        endpoint = f"datasets/{database_code}/{dataset_code}/metadata.json"

        result = self._make_request(endpoint)

        return result.get('dataset', {})

    # === Commodity Prices ===

    def get_gold_price(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30
    ) -> Dict:
        """
        Get LBMA gold prices (USD/oz)

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Number of data points

        Returns:
            Gold price data with columns: Date, USD (AM), USD (PM), GBP (AM), GBP (PM), EURO (AM), EURO (PM)

        Example:
            >>> gold = quandl.get_gold_price(limit=10)
            >>> for row in gold['data']:
            ...     date, usd_am, usd_pm = row[0], row[1], row[2]
            ...     print(f"{date}: ${usd_pm}/oz")
        """
        return self.get_dataset(
            'LBMA',
            'GOLD',
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            order='desc'
        )

    def get_silver_price(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30
    ) -> Dict:
        """
        Get LBMA silver prices (USD/oz)

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of data points

        Returns:
            Silver price data
        """
        return self.get_dataset(
            'LBMA',
            'SILVER',
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            order='desc'
        )

    def get_crude_oil_futures(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30
    ) -> Dict:
        """
        Get WTI Crude Oil Futures prices (CME)

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of data points

        Returns:
            Crude oil futures data (Open, High, Low, Last, Change, Settle, Volume, Open Interest)

        Note:
            CME data on Quandl free tier is limited. Consider using EIA API for more comprehensive oil data.
        """
        return self.get_dataset(
            'CHRIS',
            'CME_CL1',  # WTI Crude Oil Futures, Continuous Contract #1
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            order='desc'
        )

    def get_natural_gas_futures(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30
    ) -> Dict:
        """
        Get Natural Gas Futures prices (CME)

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of data points

        Returns:
            Natural gas futures data
        """
        return self.get_dataset(
            'CHRIS',
            'CME_NG1',  # Natural Gas Futures, Continuous Contract #1
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            order='desc'
        )

    def get_copper_futures(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30
    ) -> Dict:
        """
        Get Copper Futures prices (CME)

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of data points

        Returns:
            Copper futures data
        """
        return self.get_dataset(
            'CHRIS',
            'CME_HG1',  # Copper Futures, Continuous Contract #1
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            order='desc'
        )

    # === World Bank Commodity Prices ===

    def get_wb_commodity_price(
        self,
        commodity_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30
    ) -> Dict:
        """
        Get World Bank commodity price

        Args:
            commodity_code: Commodity code from World Bank
                Examples:
                - 'PALUM' - Aluminum
                - 'PCOAL' - Coal
                - 'PCOFFOTM' - Coffee
                - 'PCOTTIND' - Cotton
                - 'POILWTI' - Crude Oil WTI
                - 'POILBRE' - Crude Oil Brent
                - 'PNGASUS' - Natural Gas US
                - 'PWHEAMT' - Wheat
            start_date: Start date
            end_date: End date
            limit: Number of data points

        Returns:
            Commodity price data from World Bank

        Example:
            >>> oil = quandl.get_wb_commodity_price('POILWTI', limit=10)
        """
        return self.get_dataset(
            'ODA',
            commodity_code,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            order='desc'
        )

    # === Convenience Methods ===

    def get_all_precious_metals(self, limit: int = 10) -> Dict[str, Dict]:
        """
        Get latest prices for all precious metals

        Args:
            limit: Number of data points per metal

        Returns:
            Dictionary with gold and silver data
        """
        return {
            'gold': self.get_gold_price(limit=limit),
            'silver': self.get_silver_price(limit=limit)
        }

    def get_all_energy_commodities(self, limit: int = 10) -> Dict[str, Dict]:
        """
        Get latest prices for energy commodities

        Args:
            limit: Number of data points per commodity

        Returns:
            Dictionary with crude oil and natural gas data
        """
        try:
            crude_oil = self.get_crude_oil_futures(limit=limit)
        except:
            # Fallback to World Bank data if CME futures fail
            crude_oil = self.get_wb_commodity_price('POILWTI', limit=limit)

        try:
            nat_gas = self.get_natural_gas_futures(limit=limit)
        except:
            nat_gas = self.get_wb_commodity_price('PNGASUS', limit=limit)

        return {
            'crude_oil': crude_oil,
            'natural_gas': nat_gas
        }

    def get_commodity_summary(self) -> Dict[str, Optional[float]]:
        """
        Get latest prices for major commodities (single value per commodity)

        Returns:
            Dictionary with latest prices:
            {
                'gold_usd': 2050.5,
                'silver_usd': 24.3,
                'crude_oil_usd': 78.5,
                'natural_gas_usd': 2.8
            }
        """
        summary = {}

        # Gold
        try:
            gold = self.get_gold_price(limit=1)
            if gold.get('data') and len(gold['data']) > 0:
                # data[0] = [date, usd_am, usd_pm, ...]
                summary['gold_usd'] = gold['data'][0][2]  # USD PM price
        except:
            summary['gold_usd'] = None

        # Silver
        try:
            silver = self.get_silver_price(limit=1)
            if silver.get('data') and len(silver['data']) > 0:
                summary['silver_usd'] = silver['data'][0][1]  # USD price
        except:
            summary['silver_usd'] = None

        # Crude Oil (try futures first, fallback to World Bank)
        try:
            oil = self.get_crude_oil_futures(limit=1)
            if oil.get('data') and len(oil['data']) > 0:
                summary['crude_oil_usd'] = oil['data'][0][6]  # Settle price
        except:
            try:
                oil_wb = self.get_wb_commodity_price('POILWTI', limit=1)
                if oil_wb.get('data') and len(oil_wb['data']) > 0:
                    summary['crude_oil_usd'] = oil_wb['data'][0][1]
            except:
                summary['crude_oil_usd'] = None

        # Natural Gas
        try:
            gas = self.get_natural_gas_futures(limit=1)
            if gas.get('data') and len(gas['data']) > 0:
                summary['natural_gas_usd'] = gas['data'][0][6]  # Settle price
        except:
            try:
                gas_wb = self.get_wb_commodity_price('PNGASUS', limit=1)
                if gas_wb.get('data') and len(gas_wb['data']) > 0:
                    summary['natural_gas_usd'] = gas_wb['data'][0][1]
            except:
                summary['natural_gas_usd'] = None

        return summary


# === Singleton Pattern ===

_quandl_instance = None

def get_quandl_collector(api_key: Optional[str] = None) -> QuandlSource:
    """
    Get or create Quandl collector instance (singleton pattern)

    Args:
        api_key: Optional API key override

    Returns:
        QuandlSource instance
    """
    global _quandl_instance

    if _quandl_instance is None:
        _quandl_instance = QuandlSource(api_key=api_key)

    return _quandl_instance
