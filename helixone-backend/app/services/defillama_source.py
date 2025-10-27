"""
DefiLlama API Data Source
Documentation: https://defillama.com/docs/api

Features:
- TVL (Total Value Locked) for 2000+ protocols
- Yields/APY for DeFi protocols
- Stablecoin data
- Chain TVL (Ethereum, BSC, Polygon, etc.)
- Historical TVL data
- Protocol revenues
- 100% FREE - No API key required
- Unlimited requests

Coverage:
- 2000+ DeFi protocols
- 200+ blockchains
- 1000+ liquidity pools
- Real-time TVL updates

Use Cases:
- DeFi protocol analysis
- Yield farming opportunities
- TVL monitoring
- Chain comparison
- Protocol due diligence
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class DefiLlamaSource:
    """
    DefiLlama API collector for DeFi analytics

    Free: Unlimited public data
    Coverage: 2000+ protocols, 200+ chains
    Data: TVL, yields, stablecoins, revenues
    """

    def __init__(self):
        """Initialize DefiLlama API source"""
        self.base_url = "https://api.llama.fi"
        self.yields_url = "https://yields.llama.fi"
        self.stablecoins_url = "https://stablecoins.llama.fi"

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

    def _make_request(self, base_url: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request"""
        self._rate_limit()

        url = f"{base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"DefiLlama API request failed: {str(e)}")

    # === TVL (TOTAL VALUE LOCKED) ===

    def get_all_protocols(self) -> List[Dict]:
        """
        Get all DeFi protocols with current TVL

        Returns:
            List of protocols with TVL data

        Example:
            >>> defillama = DefiLlamaSource()
            >>> protocols = defillama.get_all_protocols()
            >>> top_10 = sorted(protocols, key=lambda x: x['tvl'], reverse=True)[:10]
            >>> for p in top_10:
            ...     print(f"{p['name']}: ${p['tvl']:,.0f}")
        """
        return self._make_request(self.base_url, "protocols")

    def get_protocol(self, slug: str) -> Dict:
        """
        Get detailed data for a specific protocol

        Args:
            slug: Protocol slug (e.g., 'aave', 'uniswap', 'curve')

        Returns:
            {
                'name': string,
                'symbol': string,
                'tvl': float,
                'chainTvls': dict,
                'change_1h': float,
                'change_1d': float,
                'change_7d': float,
                'mcap': float (market cap),
                'chains': list,
                'tvl_chart': list of historical data
            }

        Example:
            >>> aave = defillama.get_protocol('aave')
            >>> print(f"Aave TVL: ${aave['tvl']:,.0f}")
            >>> print(f"24h change: {aave['change_1d']:.2f}%")
        """
        return self._make_request(self.base_url, f"protocol/{slug}")

    def get_protocol_tvl_history(self, slug: str) -> List[Dict]:
        """
        Get historical TVL for a protocol

        Args:
            slug: Protocol slug

        Returns:
            List of {date: unix_timestamp, totalLiquidityUSD: float}
        """
        protocol = self.get_protocol(slug)
        return protocol.get('tvl', [])

    def get_current_tvl_all_protocols(self) -> Dict:
        """
        Get total TVL across all protocols

        Returns:
            {'totalTvl': float, 'timestamp': int}

        Example:
            >>> tvl = defillama.get_current_tvl_all_protocols()
            >>> print(f"Total DeFi TVL: ${tvl/1e9:.2f}B")
        """
        return self._make_request(self.base_url, "tvl")

    # === CHAINS ===

    def get_all_chains(self) -> List[Dict]:
        """
        Get all chains with TVL

        Returns:
            List of chains with TVL data

        Example:
            >>> chains = defillama.get_all_chains()
            >>> for chain in sorted(chains, key=lambda x: x['tvl'], reverse=True)[:5]:
            ...     print(f"{chain['name']}: ${chain['tvl']:,.0f}")
        """
        return self._make_request(self.base_url, "chains")

    def get_chain_tvl(self, chain: str) -> Dict:
        """
        Get TVL for a specific chain

        Args:
            chain: Chain name (e.g., 'Ethereum', 'BSC', 'Polygon')

        Returns:
            Chain TVL data with history

        Example:
            >>> eth = defillama.get_chain_tvl('Ethereum')
            >>> print(f"Ethereum TVL: ${eth['tvl']:,.0f}")
        """
        return self._make_request(self.base_url, f"chain/{chain}")

    # === YIELDS / APY ===

    def get_all_pools(self) -> List[Dict]:
        """
        Get all liquidity pools with APY

        Returns:
            List of pools with yield data

        Example:
            >>> pools = defillama.get_all_pools()
            >>> high_apy = [p for p in pools if p.get('apy', 0) > 10]
            >>> for p in sorted(high_apy, key=lambda x: x['apy'], reverse=True)[:10]:
            ...     print(f"{p['symbol']} on {p['project']}: {p['apy']:.2f}% APY")
        """
        data = self._make_request(self.yields_url, "pools")
        return data.get('data', [])

    def get_pool_by_id(self, pool_id: str) -> Dict:
        """
        Get specific pool data

        Args:
            pool_id: Pool UUID

        Returns:
            Pool data with APY, TVL, etc.
        """
        pools = self.get_all_pools()
        for pool in pools:
            if pool.get('pool') == pool_id:
                return pool
        return {}

    def get_high_yield_pools(
        self,
        min_apy: float = 10.0,
        min_tvl: float = 1000000,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get high-yield pools filtered by APY and TVL

        Args:
            min_apy: Minimum APY %
            min_tvl: Minimum TVL in USD
            limit: Max results

        Returns:
            List of high-yield pools

        Example:
            >>> pools = defillama.get_high_yield_pools(min_apy=20, min_tvl=5000000)
            >>> for p in pools[:10]:
            ...     print(f"{p['symbol']}: {p['apy']:.2f}% (TVL: ${p['tvlUsd']:,.0f})")
        """
        all_pools = self.get_all_pools()

        # Filter
        filtered = [
            p for p in all_pools
            if p.get('apy', 0) >= min_apy and p.get('tvlUsd', 0) >= min_tvl
        ]

        # Sort by APY
        sorted_pools = sorted(filtered, key=lambda x: x.get('apy', 0), reverse=True)

        return sorted_pools[:limit]

    def get_pools_by_project(self, project: str) -> List[Dict]:
        """
        Get all pools for a specific project

        Args:
            project: Project name (e.g., 'aave', 'compound', 'uniswap-v3')

        Returns:
            List of pools for that project

        Example:
            >>> aave_pools = defillama.get_pools_by_project('aave-v3')
            >>> for p in aave_pools[:5]:
            ...     print(f"{p['symbol']}: {p['apy']:.2f}% APY")
        """
        all_pools = self.get_all_pools()
        return [p for p in all_pools if p.get('project', '').lower() == project.lower()]

    def get_pools_by_chain(self, chain: str) -> List[Dict]:
        """
        Get all pools on a specific chain

        Args:
            chain: Chain name (e.g., 'Ethereum', 'Arbitrum', 'Optimism')

        Returns:
            List of pools on that chain

        Example:
            >>> arbitrum_pools = defillama.get_pools_by_chain('Arbitrum')
            >>> high_yield = sorted(arbitrum_pools, key=lambda x: x.get('apy', 0), reverse=True)[:10]
        """
        all_pools = self.get_all_pools()
        return [p for p in all_pools if p.get('chain', '').lower() == chain.lower()]

    # === STABLECOINS ===

    def get_all_stablecoins(self) -> List[Dict]:
        """
        Get all stablecoins with circulating supply

        Returns:
            List of stablecoins

        Example:
            >>> stables = defillama.get_all_stablecoins()
            >>> for s in sorted(stables, key=lambda x: x['circulating'], reverse=True)[:5]:
            ...     print(f"{s['name']}: ${s['circulating']:,.0f}")
        """
        data = self._make_request(self.stablecoins_url, "stablecoins")
        return data.get('peggedAssets', [])

    def get_stablecoin(self, stablecoin_id: int) -> Dict:
        """
        Get specific stablecoin data

        Args:
            stablecoin_id: Stablecoin ID

        Returns:
            Stablecoin data with history
        """
        return self._make_request(self.stablecoins_url, f"stablecoin/{stablecoin_id}")

    def get_stablecoins_on_chain(self, chain: str) -> Dict:
        """
        Get all stablecoins circulating on a chain

        Args:
            chain: Chain name

        Returns:
            Stablecoins on that chain
        """
        return self._make_request(self.stablecoins_url, f"stablecoinchains/{chain}")

    # === SUMMARY / CONVENIENCE ===

    def get_top_protocols(self, limit: int = 10) -> List[Dict]:
        """
        Get top protocols by TVL

        Args:
            limit: Number of protocols to return

        Returns:
            List of top protocols

        Example:
            >>> top_10 = defillama.get_top_protocols(10)
            >>> for i, p in enumerate(top_10, 1):
            ...     print(f"{i}. {p['name']}: ${p['tvl']:,.0f}")
        """
        protocols = self.get_all_protocols()
        sorted_protocols = sorted(protocols, key=lambda x: x.get('tvl', 0) or 0, reverse=True)
        return sorted_protocols[:limit]

    def get_top_chains(self, limit: int = 10) -> List[Dict]:
        """
        Get top chains by TVL

        Args:
            limit: Number of chains to return

        Returns:
            List of top chains
        """
        chains = self.get_all_chains()
        sorted_chains = sorted(chains, key=lambda x: x.get('tvl', 0) or 0, reverse=True)
        return sorted_chains[:limit]

    def get_market_overview(self) -> Dict:
        """
        Get DeFi market overview

        Returns:
            {
                'total_tvl': float,
                'top_5_protocols': list,
                'top_5_chains': list,
                'top_5_yields': list,
                'timestamp': datetime
            }

        Example:
            >>> overview = defillama.get_market_overview()
            >>> print(f"Total DeFi TVL: ${overview['total_tvl']/1e9:.2f}B")
        """
        total_tvl = self._make_request(self.base_url, "tvl")
        top_protocols = self.get_top_protocols(5)
        top_chains = self.get_top_chains(5)
        top_yields = self.get_high_yield_pools(min_apy=5, min_tvl=1000000, limit=5)

        return {
            'total_tvl': total_tvl,
            'top_5_protocols': [
                {'name': p['name'], 'tvl': p['tvl']}
                for p in top_protocols
            ],
            'top_5_chains': [
                {'name': c['name'], 'tvl': c['tvl']}
                for c in top_chains
            ],
            'top_5_yields': [
                {
                    'symbol': p.get('symbol', ''),
                    'project': p.get('project', ''),
                    'apy': p.get('apy', 0),
                    'tvl': p.get('tvlUsd', 0)
                }
                for p in top_yields
            ],
            'timestamp': datetime.now()
        }


# === SINGLETON PATTERN ===

_defillama_instance = None

def get_defillama_collector() -> DefiLlamaSource:
    """
    Get or create DefiLlama collector instance (singleton)

    Returns:
        DefiLlamaSource instance
    """
    global _defillama_instance

    if _defillama_instance is None:
        _defillama_instance = DefiLlamaSource()

    return _defillama_instance
