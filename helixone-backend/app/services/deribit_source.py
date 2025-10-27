"""
Deribit API Data Source - Crypto Options & Futures
Documentation: https://docs.deribit.com/

Features:
- Crypto options (BTC, ETH, SOL)
- Greeks pre-calculated (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility
- Open interest
- Historical volatility
- Option chains
- Futures data
- 100% FREE - No API key required

Coverage:
- Bitcoin (BTC) options & futures
- Ethereum (ETH) options & futures
- Solana (SOL) options & futures

Use Cases:
- Options trading strategies
- Volatility analysis
- Greeks-based hedging
- Put/Call ratio analysis
- Max pain analysis
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal


class DeribitSource:
    """
    Deribit API collector for crypto options and derivatives

    Free: Unlimited public data (no auth required)
    Coverage: BTC, ETH, SOL options & futures
    Data: Greeks, IV, OI, option chains
    """

    def __init__(self):
        """Initialize Deribit API source"""
        self.base_url = "https://www.deribit.com/api/v2/public"

        # Rate limiting (be respectful)
        self.min_request_interval = 0.1  # 10 req/sec max
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request"""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('error'):
                raise Exception(f"Deribit API error: {data['error']}")

            return data.get('result', {})

        except requests.exceptions.RequestException as e:
            raise Exception(f"Deribit API request failed: {str(e)}")

    # === INSTRUMENTS ===

    def get_instruments(
        self,
        currency: str = 'BTC',
        kind: str = 'option',
        expired: bool = False
    ) -> List[Dict]:
        """
        Get all instruments for a currency

        Args:
            currency: 'BTC', 'ETH', or 'SOL'
            kind: 'option', 'future', 'spot', 'option_combo', 'future_combo'
            expired: Include expired instruments

        Returns:
            List of instruments

        Example:
            >>> deribit = DeribitSource()
            >>> options = deribit.get_instruments('BTC', 'option')
            >>> print(f"Found {len(options)} BTC options")
        """
        params = {
            'currency': currency.upper(),
            'kind': kind,
            'expired': 'true' if expired else 'false'  # API requires string
        }

        return self._make_request('get_instruments', params)

    def get_instrument(self, instrument_name: str) -> Dict:
        """
        Get single instrument details

        Args:
            instrument_name: e.g., 'BTC-27DEC24-100000-C'

        Returns:
            Instrument details with Greeks
        """
        params = {'instrument_name': instrument_name}
        return self._make_request('get_instrument', params)

    # === OPTION CHAIN ===

    def get_option_chain(
        self,
        currency: str = 'BTC',
        expiration: Optional[str] = None
    ) -> List[Dict]:
        """
        Get full option chain for a currency

        Args:
            currency: 'BTC', 'ETH', or 'SOL'
            expiration: Expiration date (e.g., '27DEC24') or None for all

        Returns:
            List of options with prices and Greeks

        Example:
            >>> chain = deribit.get_option_chain('BTC', '27DEC24')
            >>> for opt in chain[:5]:
            ...     print(f"{opt['instrument_name']}: IV={opt['mark_iv']}%")
        """
        options = self.get_instruments(currency, 'option')

        if expiration:
            options = [opt for opt in options if expiration.upper() in opt['instrument_name']]

        # Enrich with market data
        enriched = []
        for opt in options:
            try:
                data = self.get_ticker(opt['instrument_name'])
                opt.update(data)
                enriched.append(opt)
            except:
                pass

        return enriched

    # === TICKER & GREEKS ===

    def get_ticker(self, instrument_name: str) -> Dict:
        """
        Get ticker with Greeks for an instrument

        Args:
            instrument_name: e.g., 'BTC-27DEC24-100000-C'

        Returns:
            {
                'last_price': float,
                'mark_price': float,
                'mark_iv': float (implied volatility %),
                'bid_price': float,
                'ask_price': float,
                'open_interest': float,
                'volume_24h': float,
                'greeks': {
                    'delta': float,
                    'gamma': float,
                    'theta': float,
                    'vega': float,
                    'rho': float
                },
                'underlying_price': float,
                'underlying_index': string
            }

        Example:
            >>> ticker = deribit.get_ticker('BTC-27DEC24-100000-C')
            >>> print(f"Delta: {ticker['greeks']['delta']:.4f}")
            >>> print(f"IV: {ticker['mark_iv']:.2f}%")
        """
        params = {'instrument_name': instrument_name}
        return self._make_request('ticker', params)

    def get_multiple_tickers(self, instrument_names: List[str]) -> List[Dict]:
        """Get tickers for multiple instruments"""
        tickers = []
        for name in instrument_names:
            try:
                ticker = self.get_ticker(name)
                ticker['instrument_name'] = name
                tickers.append(ticker)
            except:
                pass

        return tickers

    # === ORDERBOOK ===

    def get_orderbook(self, instrument_name: str, depth: int = 10) -> Dict:
        """
        Get orderbook for an instrument

        Args:
            instrument_name: e.g., 'BTC-27DEC24-100000-C'
            depth: Number of levels (default 10)

        Returns:
            {
                'bids': [[price, quantity], ...],
                'asks': [[price, quantity], ...],
                'best_bid_price': float,
                'best_ask_price': float,
                'mark_price': float,
                'last_price': float
            }
        """
        params = {
            'instrument_name': instrument_name,
            'depth': depth
        }
        return self._make_request('get_order_book', params)

    # === EXPIRATIONS ===

    def get_expirations(self, currency: str = 'BTC') -> List[str]:
        """
        Get all available expiration dates

        Args:
            currency: 'BTC', 'ETH', or 'SOL'

        Returns:
            List of expiration dates (e.g., ['27DEC24', '3JAN25', ...])

        Example:
            >>> expirations = deribit.get_expirations('BTC')
            >>> print(f"Next expiration: {expirations[0]}")
        """
        options = self.get_instruments(currency, 'option')

        # Extract unique expirations from instrument names
        # Format: BTC-27DEC24-100000-C
        expirations = set()
        for opt in options:
            parts = opt['instrument_name'].split('-')
            if len(parts) >= 2:
                expirations.add(parts[1])

        # Sort by date
        return sorted(list(expirations))

    # === STRIKES ===

    def get_strikes(
        self,
        currency: str = 'BTC',
        expiration: Optional[str] = None
    ) -> List[float]:
        """
        Get all available strikes

        Args:
            currency: 'BTC', 'ETH', or 'SOL'
            expiration: Filter by expiration (optional)

        Returns:
            Sorted list of strike prices

        Example:
            >>> strikes = deribit.get_strikes('BTC', '27DEC24')
            >>> print(f"ATM strike: ${strikes[len(strikes)//2]:,.0f}")
        """
        options = self.get_instruments(currency, 'option')

        if expiration:
            options = [opt for opt in options if expiration.upper() in opt['instrument_name']]

        # Extract strikes from instrument names
        strikes = set()
        for opt in options:
            strikes.add(opt['strike'])

        return sorted(list(strikes))

    # === VOLATILITY ===

    def get_volatility_index(self, currency: str = 'BTC') -> Dict:
        """
        Get volatility index (like VIX for crypto)

        Args:
            currency: 'BTC', 'ETH', or 'SOL'

        Returns:
            {
                'index_name': string,
                'price': float (volatility in %)
            }

        Example:
            >>> vol = deribit.get_volatility_index('BTC')
            >>> print(f"BTC volatility: {vol['price']:.2f}%")
        """
        # Deribit uses dvol format
        index_name = f"dvol_{currency.lower()}_usd"
        params = {'index_name': index_name}

        try:
            return self._make_request('get_index_price', params)
        except:
            # Fallback if vol index not available
            return {'index_name': index_name, 'index_price': 0}

    def get_historical_volatility(
        self,
        currency: str = 'BTC'
    ) -> Dict:
        """
        Get historical volatility

        Args:
            currency: 'BTC', 'ETH', or 'SOL'

        Returns:
            Historical volatility data
        """
        return self.get_volatility_index(currency)

    # === PUT/CALL ANALYSIS ===

    def get_put_call_ratio(
        self,
        currency: str = 'BTC',
        expiration: Optional[str] = None
    ) -> Dict:
        """
        Calculate Put/Call ratio from open interest

        Args:
            currency: 'BTC', 'ETH', or 'SOL'
            expiration: Filter by expiration (optional)

        Returns:
            {
                'put_oi': float,
                'call_oi': float,
                'ratio': float (put_oi / call_oi),
                'total_oi': float
            }

        Example:
            >>> pc = deribit.get_put_call_ratio('BTC', '27DEC24')
            >>> print(f"P/C Ratio: {pc['ratio']:.2f}")
            >>> if pc['ratio'] > 1:
            ...     print("More puts than calls (bearish)")
        """
        options = self.get_option_chain(currency, expiration)

        put_oi = 0
        call_oi = 0

        for opt in options:
            oi = opt.get('open_interest', 0)

            if opt['instrument_name'].endswith('-P'):  # Put
                put_oi += oi
            elif opt['instrument_name'].endswith('-C'):  # Call
                call_oi += oi

        ratio = put_oi / call_oi if call_oi > 0 else 0

        return {
            'put_oi': put_oi,
            'call_oi': call_oi,
            'ratio': ratio,
            'total_oi': put_oi + call_oi,
            'currency': currency,
            'expiration': expiration
        }

    # === SPOT PRICE ===

    def get_spot_price(self, currency: str = 'BTC') -> float:
        """
        Get current spot price

        Args:
            currency: 'BTC', 'ETH', or 'SOL'

        Returns:
            Current spot price

        Example:
            >>> btc_price = deribit.get_spot_price('BTC')
            >>> print(f"BTC: ${btc_price:,.2f}")
        """
        index_name = f"{currency.lower()}_usd"
        params = {'index_name': index_name}

        result = self._make_request('get_index_price', params)
        return result.get('index_price', 0)

    # === CONVENIENCE METHODS ===

    def get_atm_options(
        self,
        currency: str = 'BTC',
        expiration: Optional[str] = None
    ) -> Dict:
        """
        Get at-the-money call and put

        Args:
            currency: 'BTC', 'ETH', or 'SOL'
            expiration: Expiration date (e.g., '27DEC24')

        Returns:
            {
                'spot_price': float,
                'atm_strike': float,
                'call': dict with ticker data,
                'put': dict with ticker data
            }

        Example:
            >>> atm = deribit.get_atm_options('BTC', '27DEC24')
            >>> print(f"ATM Call IV: {atm['call']['mark_iv']:.2f}%")
            >>> print(f"ATM Put IV: {atm['put']['mark_iv']:.2f}%")
        """
        spot = self.get_spot_price(currency)
        strikes = self.get_strikes(currency, expiration)

        # Find closest strike to spot
        atm_strike = min(strikes, key=lambda x: abs(x - spot))

        # Build instrument names
        if not expiration:
            expirations = self.get_expirations(currency)
            expiration = expirations[0] if expirations else None

        call_name = f"{currency.upper()}-{expiration}-{int(atm_strike)}-C"
        put_name = f"{currency.upper()}-{expiration}-{int(atm_strike)}-P"

        return {
            'spot_price': spot,
            'atm_strike': atm_strike,
            'call': self.get_ticker(call_name),
            'put': self.get_ticker(put_name),
            'expiration': expiration
        }

    def get_option_summary(self, currency: str = 'BTC') -> Dict:
        """
        Get summary of options market

        Args:
            currency: 'BTC', 'ETH', or 'SOL'

        Returns:
            {
                'currency': string,
                'spot_price': float,
                'volatility_index': float,
                'total_options': int,
                'expirations': list,
                'put_call_ratio': float,
                'total_open_interest': float
            }
        """
        spot = self.get_spot_price(currency)
        vol = self.get_volatility_index(currency)
        expirations = self.get_expirations(currency)
        options = self.get_instruments(currency, 'option')
        pc_ratio = self.get_put_call_ratio(currency)

        return {
            'currency': currency,
            'spot_price': spot,
            'volatility_index': vol.get('index_price', 0),
            'total_options': len(options),
            'expirations': expirations[:5],  # First 5
            'put_call_ratio': pc_ratio['ratio'],
            'total_open_interest': pc_ratio['total_oi']
        }


# === SINGLETON PATTERN ===

_deribit_instance = None

def get_deribit_collector() -> DeribitSource:
    """
    Get or create Deribit collector instance (singleton)

    Returns:
        DeribitSource instance
    """
    global _deribit_instance

    if _deribit_instance is None:
        _deribit_instance = DeribitSource()

    return _deribit_instance
