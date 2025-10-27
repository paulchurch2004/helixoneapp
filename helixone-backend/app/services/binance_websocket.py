"""
Binance WebSocket Data Source
Documentation: https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams

Features:
- Real-time orderbook (depth) streaming
- Trade stream (live executions)
- Kline/candlestick streams
- Ticker streams (24h stats)
- All streams FREE and UNLIMITED

WebSocket Endpoints:
- wss://stream.binance.com:9443/ws/<streamName>
- wss://stream.binance.com:9443/stream?streams=<streamName1>/<streamName2>

Use Cases:
- Market making (orderbook depth)
- Arbitrage (price differences)
- Scalping (tick-by-tick data)
- Liquidity analysis
- Real-time charting
"""

import json
import asyncio
import websockets
from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """
    Binance WebSocket client for real-time market data

    Features:
    - Orderbook depth streaming (20 levels, 100ms updates)
    - Trade streaming (every execution)
    - Kline streaming (real-time candles)
    - Ticker streaming (24h stats)
    """

    def __init__(self):
        """Initialize Binance WebSocket client"""
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.stream_url = "wss://stream.binance.com:9443/stream"
        self.websocket = None
        self.running = False

    # === ORDERBOOK DEPTH STREAM ===

    async def stream_orderbook(
        self,
        symbol: str,
        callback: Callable,
        levels: int = 20,
        update_speed: str = "100ms"
    ):
        """
        Stream real-time orderbook depth

        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            callback: Function to call with orderbook updates
            levels: Depth levels (5, 10, 20) - default 20
            update_speed: Update frequency ('100ms' or '1000ms')

        Example:
            >>> async def handle_orderbook(data):
            ...     bids = data['bids']  # [[price, quantity], ...]
            ...     asks = data['asks']  # [[price, quantity], ...]
            ...     print(f"Best bid: {bids[0][0]}, Best ask: {asks[0][0]}")
            ...
            >>> ws = BinanceWebSocket()
            >>> await ws.stream_orderbook('btcusdt', handle_orderbook)
        """
        symbol = symbol.lower()

        # Stream name: <symbol>@depth<levels>@<speed>
        # e.g., btcusdt@depth20@100ms
        if update_speed == "100ms":
            stream = f"{symbol}@depth{levels}@100ms"
        else:
            stream = f"{symbol}@depth{levels}"

        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "orderbook")

    async def stream_orderbook_diff(self, symbol: str, callback: Callable):
        """
        Stream orderbook differential updates (faster, more data)

        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            callback: Function to call with diff updates

        Note:
        - Updates every 100ms
        - Contains only changes (not full book)
        - More efficient for bandwidth

        Example:
            >>> async def handle_diff(data):
            ...     bids = data['b']  # Updated bids
            ...     asks = data['a']  # Updated asks
            ...     print(f"Updated {len(bids)} bids, {len(asks)} asks")
            ...
            >>> await ws.stream_orderbook_diff('btcusdt', handle_diff)
        """
        symbol = symbol.lower()
        stream = f"{symbol}@depth"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "orderbook_diff")

    # === TRADE STREAM ===

    async def stream_trades(self, symbol: str, callback: Callable):
        """
        Stream individual trades in real-time

        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            callback: Function to call with each trade

        Example:
            >>> async def handle_trade(data):
            ...     price = float(data['p'])
            ...     quantity = float(data['q'])
            ...     is_buyer_maker = data['m']
            ...     print(f"Trade: {quantity} @ ${price} ({'SELL' if is_buyer_maker else 'BUY'})")
            ...
            >>> await ws.stream_trades('btcusdt', handle_trade)
        """
        symbol = symbol.lower()
        stream = f"{symbol}@trade"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "trades")

    # === KLINE (CANDLESTICK) STREAM ===

    async def stream_klines(
        self,
        symbol: str,
        interval: str,
        callback: Callable
    ):
        """
        Stream real-time kline/candlestick data

        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d')
            callback: Function to call with kline updates

        Example:
            >>> async def handle_kline(data):
            ...     k = data['k']
            ...     print(f"Candle: O={k['o']} H={k['h']} L={k['l']} C={k['c']}")
            ...
            >>> await ws.stream_klines('btcusdt', '1m', handle_kline)
        """
        symbol = symbol.lower()
        stream = f"{symbol}@kline_{interval}"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "klines")

    # === TICKER STREAM ===

    async def stream_ticker(self, symbol: str, callback: Callable):
        """
        Stream 24h ticker statistics

        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            callback: Function to call with ticker updates

        Example:
            >>> async def handle_ticker(data):
            ...     print(f"Price: ${data['c']}, 24h Vol: {data['v']}")
            ...
            >>> await ws.stream_ticker('btcusdt', handle_ticker)
        """
        symbol = symbol.lower()
        stream = f"{symbol}@ticker"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "ticker")

    async def stream_mini_ticker(self, symbol: str, callback: Callable):
        """
        Stream mini ticker (lighter version, 1000ms updates)

        Args:
            symbol: Trading pair
            callback: Function to call with mini ticker
        """
        symbol = symbol.lower()
        stream = f"{symbol}@miniTicker"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "mini_ticker")

    # === ALL MARKET TICKERS ===

    async def stream_all_tickers(self, callback: Callable):
        """
        Stream all market tickers (all trading pairs)

        Args:
            callback: Function to call with all tickers array

        Warning: High bandwidth - use mini ticker for production
        """
        stream = "!ticker@arr"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "all_tickers")

    async def stream_all_mini_tickers(self, callback: Callable):
        """
        Stream all mini tickers (lighter, recommended)

        Args:
            callback: Function to call with mini tickers array
        """
        stream = "!miniTicker@arr"
        url = f"{self.base_url}/{stream}"

        await self._connect_and_stream(url, callback, "all_mini_tickers")

    # === COMBINED STREAMS ===

    async def stream_combined(
        self,
        streams: List[str],
        callback: Callable
    ):
        """
        Subscribe to multiple streams simultaneously

        Args:
            streams: List of stream names
            callback: Function to call with stream data

        Example:
            >>> streams = [
            ...     'btcusdt@depth20@100ms',
            ...     'btcusdt@trade',
            ...     'ethusdt@depth20@100ms'
            ... ]
            >>> await ws.stream_combined(streams, handle_data)
        """
        streams_param = '/'.join(streams)
        url = f"{self.stream_url}?streams={streams_param}"

        await self._connect_and_stream(url, callback, "combined")

    # === CORE WEBSOCKET LOGIC ===

    async def _connect_and_stream(
        self,
        url: str,
        callback: Callable,
        stream_type: str
    ):
        """
        Connect to WebSocket and stream data

        Args:
            url: WebSocket URL
            callback: Callback function
            stream_type: Type of stream (for logging)
        """
        self.running = True
        retry_delay = 1
        max_retry_delay = 60

        while self.running:
            try:
                logger.info(f"🔌 Connecting to Binance {stream_type} stream...")

                async with websockets.connect(url, ping_interval=20) as websocket:
                    self.websocket = websocket
                    logger.info(f"✅ Connected to {stream_type} stream")
                    retry_delay = 1  # Reset retry delay on successful connection

                    # Receive messages
                    async for message in websocket:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)

                            # Handle combined stream format
                            if 'stream' in data:
                                data = data['data']

                            # Call user callback
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data)
                            else:
                                callback(data)

                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"❌ Callback error: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"⚠️  Connection closed, reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

        logger.info(f"🔌 {stream_type} stream stopped")

    def stop(self):
        """Stop the WebSocket stream"""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())

    # === HELPER METHODS ===

    @staticmethod
    def parse_orderbook(data: Dict) -> Dict:
        """
        Parse orderbook data to friendly format

        Returns:
            {
                'bids': [(price, quantity), ...],
                'asks': [(price, quantity), ...],
                'timestamp': datetime
            }
        """
        return {
            'bids': [(float(p), float(q)) for p, q in data.get('bids', [])],
            'asks': [(float(p), float(q)) for p, q in data.get('asks', [])],
            'timestamp': datetime.fromtimestamp(data.get('E', 0) / 1000)
        }

    @staticmethod
    def parse_trade(data: Dict) -> Dict:
        """
        Parse trade data to friendly format

        Returns:
            {
                'price': float,
                'quantity': float,
                'timestamp': datetime,
                'is_sell': bool,
                'trade_id': int
            }
        """
        return {
            'price': float(data.get('p', 0)),
            'quantity': float(data.get('q', 0)),
            'timestamp': datetime.fromtimestamp(data.get('T', 0) / 1000),
            'is_sell': data.get('m', False),  # True if buyer is market maker (sell)
            'trade_id': data.get('t', 0)
        }

    @staticmethod
    def parse_kline(data: Dict) -> Dict:
        """
        Parse kline data to friendly format

        Returns:
            {
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float,
                'timestamp': datetime,
                'is_closed': bool
            }
        """
        k = data.get('k', {})
        return {
            'open': float(k.get('o', 0)),
            'high': float(k.get('h', 0)),
            'low': float(k.get('l', 0)),
            'close': float(k.get('c', 0)),
            'volume': float(k.get('v', 0)),
            'timestamp': datetime.fromtimestamp(k.get('t', 0) / 1000),
            'is_closed': k.get('x', False)
        }


# === SINGLETON PATTERN ===

_binance_ws_instance = None

def get_binance_websocket() -> BinanceWebSocket:
    """
    Get or create Binance WebSocket instance (singleton)

    Returns:
        BinanceWebSocket instance
    """
    global _binance_ws_instance

    if _binance_ws_instance is None:
        _binance_ws_instance = BinanceWebSocket()

    return _binance_ws_instance
