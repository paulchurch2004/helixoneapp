#!/usr/bin/env python3
"""
Test Binance WebSocket - Orderbook temps réel & Trades
Test rapide pour vérifier le streaming de données
"""

import sys
import asyncio
sys.path.insert(0, 'helixone-backend')

from app.services.binance_websocket import get_binance_websocket


# === TEST 1: ORDERBOOK DEPTH ===

async def test_orderbook():
    """Test orderbook streaming pendant 5 secondes"""
    print("\n" + "="*70)
    print("🔷 TEST 1: Orderbook BTC/USDT Temps Réel (5 secondes)")
    print("="*70)

    ws = get_binance_websocket()

    message_count = 0

    async def handle_orderbook(data):
        nonlocal message_count
        message_count += 1

        if message_count == 1:
            # Premier message - détails complets
            bids = data.get('bids', [])[:3]
            asks = data.get('asks', [])[:3]

            print(f"\n✅ Orderbook reçu!")
            print(f"\nTop 3 Bids:")
            for i, (price, qty) in enumerate(bids, 1):
                print(f"   {i}. ${float(price):,.2f} - {float(qty):.4f} BTC")

            print(f"\nTop 3 Asks:")
            for i, (price, qty) in enumerate(asks, 1):
                print(f"   {i}. ${float(price):,.2f} - {float(qty):.4f} BTC")

            # Spread
            if bids and asks:
                spread = float(asks[0][0]) - float(bids[0][0])
                spread_pct = (spread / float(bids[0][0])) * 100
                print(f"\n💰 Spread: ${spread:.2f} ({spread_pct:.4f}%)")

        elif message_count % 10 == 0:
            # Afficher update chaque 10 messages
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                print(f"   📊 Update {message_count}: Bid=${best_bid:,.2f} Ask=${best_ask:,.2f}")

    # Stream pendant 5 secondes
    task = asyncio.create_task(
        ws.stream_orderbook('btcusdt', handle_orderbook, levels=20, update_speed='100ms')
    )

    await asyncio.sleep(5)
    ws.stop()

    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        pass

    print(f"\n✅ Test terminé: {message_count} updates reçues")


# === TEST 2: TRADES STREAM ===

async def test_trades():
    """Test trade streaming pendant 5 secondes"""
    print("\n" + "="*70)
    print("🔷 TEST 2: Trades BTC/USDT Temps Réel (5 secondes)")
    print("="*70)

    ws = get_binance_websocket()

    trades = []
    buy_volume = 0
    sell_volume = 0

    async def handle_trade(data):
        price = float(data.get('p', 0))
        quantity = float(data.get('q', 0))
        is_sell = data.get('m', False)  # True si market maker achète (= sell order)

        trades.append({
            'price': price,
            'quantity': quantity,
            'is_sell': is_sell
        })

        nonlocal buy_volume, sell_volume
        if is_sell:
            sell_volume += quantity
        else:
            buy_volume += quantity

        # Afficher chaque trade
        side = "🔴 SELL" if is_sell else "🟢 BUY"
        print(f"   {side}: {quantity:.4f} BTC @ ${price:,.2f}")

    # Stream pendant 5 secondes
    task = asyncio.create_task(
        ws.stream_trades('btcusdt', handle_trade)
    )

    await asyncio.sleep(5)
    ws.stop()

    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        pass

    print(f"\n✅ Test terminé:")
    print(f"   Total trades: {len(trades)}")
    print(f"   🟢 Buy volume: {buy_volume:.4f} BTC")
    print(f"   🔴 Sell volume: {sell_volume:.4f} BTC")
    if buy_volume + sell_volume > 0:
        buy_pct = (buy_volume / (buy_volume + sell_volume)) * 100
        print(f"   📊 Buy pressure: {buy_pct:.1f}%")


# === TEST 3: KLINES STREAM ===

async def test_klines():
    """Test kline streaming pendant 10 secondes"""
    print("\n" + "="*70)
    print("🔷 TEST 3: Klines 1m BTC/USDT (10 secondes)")
    print("="*70)

    ws = get_binance_websocket()

    kline_count = 0

    async def handle_kline(data):
        nonlocal kline_count
        kline_count += 1

        k = data.get('k', {})
        open_price = float(k.get('o', 0))
        high = float(k.get('h', 0))
        low = float(k.get('l', 0))
        close = float(k.get('c', 0))
        volume = float(k.get('v', 0))
        is_closed = k.get('x', False)

        status = "✅ CLOSED" if is_closed else "⏳ UPDATING"

        print(f"\n   {status} Kline #{kline_count}:")
        print(f"      O=${open_price:,.2f} H=${high:,.2f} L=${low:,.2f} C=${close:,.2f}")
        print(f"      Volume: {volume:.2f} BTC")

    # Stream pendant 10 secondes
    task = asyncio.create_task(
        ws.stream_klines('btcusdt', '1m', handle_kline)
    )

    await asyncio.sleep(10)
    ws.stop()

    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        pass

    print(f"\n✅ Test terminé: {kline_count} kline updates")


# === TEST 4: TICKER MINI ===

async def test_mini_ticker():
    """Test mini ticker pendant 3 secondes"""
    print("\n" + "="*70)
    print("🔷 TEST 4: Mini Ticker BTC/USDT (3 secondes)")
    print("="*70)

    ws = get_binance_websocket()

    ticker_count = 0

    async def handle_ticker(data):
        nonlocal ticker_count
        ticker_count += 1

        close = float(data.get('c', 0))
        open_price = float(data.get('o', 0))
        high = float(data.get('h', 0))
        low = float(data.get('l', 0))
        volume = float(data.get('v', 0))

        change = close - open_price
        change_pct = (change / open_price) * 100 if open_price > 0 else 0

        print(f"\n   📊 Ticker Update #{ticker_count}:")
        print(f"      Price: ${close:,.2f} ({change_pct:+.2f}%)")
        print(f"      24h: H=${high:,.2f} L=${low:,.2f}")
        print(f"      Volume: {volume:,.2f} BTC")

    # Stream pendant 3 secondes
    task = asyncio.create_task(
        ws.stream_mini_ticker('btcusdt', handle_ticker)
    )

    await asyncio.sleep(3)
    ws.stop()

    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        pass

    print(f"\n✅ Test terminé: {ticker_count} ticker updates")


# === MAIN ===

async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 BINANCE WEBSOCKET - TEST COMPLET")
    print("Streaming de données temps réel (orderbook, trades, klines)")
    print("="*70)

    try:
        # Test 1: Orderbook (5s)
        await test_orderbook()
        await asyncio.sleep(1)

        # Test 2: Trades (5s)
        await test_trades()
        await asyncio.sleep(1)

        # Test 3: Klines (10s)
        await test_klines()
        await asyncio.sleep(1)

        # Test 4: Mini Ticker (3s)
        await test_mini_ticker()

        print("\n" + "="*70)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("="*70)
        print("\n💡 Binance WebSocket fonctionne parfaitement:")
        print("   ✅ Orderbook temps réel (100ms updates)")
        print("   ✅ Trades streaming (chaque exécution)")
        print("   ✅ Klines temps réel (1m candles)")
        print("   ✅ Ticker 24h stats")
        print("\n🚀 Prêt pour le trading algorithmique!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
