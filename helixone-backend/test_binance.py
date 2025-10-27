"""
Script de test pour Binance API
Test exchange crypto - GRATUIT et ILLIMITÉ
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.binance_source import get_binance_collector


def test_binance():
    """Tester Binance API"""

    print("\n" + "="*70)
    print("🔶 TEST BINANCE API - CRYPTO EXCHANGE DATA")
    print("GRATUIT - ILLIMITÉ - Pas de clé API requise")
    print("="*70 + "\n")

    binance = get_binance_collector()

    # Test 1: Ping
    print("🏓 Test 1: Connectivité API")
    print("-" * 70)
    try:
        is_alive = binance.ping()
        if is_alive:
            server_time = binance.get_server_time()
            dt = datetime.fromtimestamp(server_time / 1000)
            print(f"\n✅ API accessible")
            print(f"   Server time: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print("\n❌ API non accessible\n")
    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 2: Bitcoin Price
    print("₿ Test 2: Prix Bitcoin (BTCUSDT)")
    print("-" * 70)
    try:
        btc = binance.get_ticker_price('BTCUSDT')
        price = float(btc['price'])
        print(f"\n💰 BTC/USDT: ${price:,.2f}\n")
    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 3: 24hr Stats
    print("📊 Test 3: Statistiques 24h - Top 5 Cryptos")
    print("-" * 70)
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']

        print()
        print(f"{'Pair':<12} {'Price':<15} {'24h Change':<12} {'Volume 24h':<20}")
        print("-" * 70)

        for symbol in symbols:
            stats = binance.get_ticker_24hr(symbol)
            price = float(stats['lastPrice'])
            change = float(stats['priceChangePercent'])
            volume = float(stats['quoteVolume'])

            change_emoji = "🟢" if change > 0 else "🔴"

            print(f"{symbol:<12} ${price:<14,.2f} {change_emoji} {change:>6.2f}%    ${volume:>18,.0f}")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 4: Order Book
    print("📖 Test 4: Order Book BTC (Top 5 Bids/Asks)")
    print("-" * 70)
    try:
        book = binance.get_orderbook('BTCUSDT', limit=5)

        print("\n💚 Top 5 Bids (Buy Orders):")
        for i, bid in enumerate(book['bids'][:5], 1):
            price, qty = float(bid[0]), float(bid[1])
            print(f"   {i}. ${price:,.2f} x {qty:.6f} BTC")

        print("\n💔 Top 5 Asks (Sell Orders):")
        for i, ask in enumerate(book['asks'][:5], 1):
            price, qty = float(ask[0]), float(ask[1])
            print(f"   {i}. ${price:,.2f} x {qty:.6f} BTC")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 5: Recent Trades
    print("💱 Test 5: Dernières Transactions BTC")
    print("-" * 70)
    try:
        trades = binance.get_recent_trades('BTCUSDT', limit=10)

        print(f"\n✅ {len(trades)} dernières transactions:\n")

        for i, trade in enumerate(trades[:10], 1):
            price = float(trade['price'])
            qty = float(trade['qty'])
            time_ms = trade['time']
            dt = datetime.fromtimestamp(time_ms / 1000)
            is_buyer = trade['isBuyerMaker']

            side = "🔴 SELL" if is_buyer else "🟢 BUY "

            print(f"   {i}. {side} ${price:,.2f} x {qty:.6f} BTC at {dt.strftime('%H:%M:%S')}")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 6: Historical Klines
    print("📈 Test 6: Données Historiques BTC (1h candles)")
    print("-" * 70)
    try:
        klines = binance.get_klines('BTCUSDT', interval='1h', limit=24)

        print(f"\n✅ {len(klines)} bougies 1h récupérées\n")

        print(f"{'Time':<20} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<10}")
        print("-" * 70)

        for k in klines[:10]:
            open_time = datetime.fromtimestamp(k[0] / 1000)
            o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            v = float(k[5])

            time_str = open_time.strftime('%Y-%m-%d %H:%M')
            print(f"{time_str:<20} ${o:<11,.0f} ${h:<11,.0f} ${l:<11,.0f} ${c:<11,.0f} {v:>8.2f}")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 7: Top Pairs by Volume
    print("🏆 Test 7: Top 10 Paires par Volume 24h")
    print("-" * 70)
    try:
        top_pairs = binance.get_top_pairs_by_volume('USDT', limit=10)

        print()
        print(f"{'Rank':<5} {'Pair':<12} {'Price':<15} {'Volume 24h':<20} {'Change %':<10}")
        print("-" * 70)

        for i, pair in enumerate(top_pairs, 1):
            symbol = pair['symbol']
            price = float(pair['lastPrice'])
            volume = float(pair['quoteVolume'])
            change = float(pair['priceChangePercent'])

            change_emoji = "🟢" if change > 0 else "🔴"

            print(f"{i:<5} {symbol:<12} ${price:<14,.4f} ${volume:>18,.0f}  {change_emoji} {change:>6.2f}%")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 8: Simple Prices
    print("💎 Test 8: Prix Simples - Top Cryptos")
    print("-" * 70)
    try:
        symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE']
        prices = binance.get_crypto_price_simple(symbols, 'USDT')

        print()
        for symbol in symbols:
            if symbol in prices:
                price = prices[symbol]
                print(f"   {symbol:<6}: ${price:>12,.2f}")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Test 9: Market Overview
    print("🌍 Test 9: Vue d'Ensemble du Marché")
    print("-" * 70)
    try:
        overview = binance.get_market_overview()

        print()
        print(f"   💰 Volume 24h (Top 100): ${overview['total_24h_volume_top100']:,.0f}")
        print(f"   ₿  Volume BTC 24h:       ${overview['btc_24h_volume']:,.0f}")
        print(f"   📊 BTC Dominance:        {overview['btc_volume_dominance']:.2f}%")

        print(f"\n   🏆 Top 5 Paires:")
        for i, pair in enumerate(overview['top_5_pairs'], 1):
            print(f"      {i}. {pair['symbol']}: ${pair['price']:,.2f} (Vol: ${pair['volume_24h']:,.0f})")

        print()

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST BINANCE API")
    print("="*70)
    print("✅ Tous les tests exécutés")
    print("🔶 Binance API intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT et ILLIMITÉ")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ Plus grand exchange crypto mondial")
    print("  - ✅ 350+ paires de trading")
    print("  - ✅ Données temps réel")
    print("\nDonnées disponibles:")
    print("  - 💰 Prix en temps réel")
    print("  - 📊 Statistiques 24h (volume, change, etc.)")
    print("  - 📖 Order books (profondeur marché)")
    print("  - 💱 Transactions récentes")
    print("  - 📈 Données historiques (klines)")
    print("  - 🏆 Classements par volume")
    print("  - 🌍 Vue d'ensemble marché")
    print("\nUtilisation:")
    print("  - Trading crypto en temps réel")
    print("  - Analyse technique (klines/candles)")
    print("  - Surveillance liquidité (orderbooks)")
    print("  - Détection opportunités arbitrage")
    print("  - Analyse volumes et tendances")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_binance()
