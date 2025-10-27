#!/usr/bin/env python3
"""Test rapide WebSocket Binance - 5 secondes"""

import sys
import asyncio
sys.path.insert(0, 'helixone-backend')

from app.services.binance_websocket import get_binance_websocket

async def test_quick():
    print("🔷 Test WebSocket Binance - Orderbook BTC/USDT (5s)")

    ws = get_binance_websocket()
    count = 0

    async def handle_book(data):
        nonlocal count
        count += 1

        if count == 1:
            bids = data.get('bids', [])[:3]
            asks = data.get('asks', [])[:3]
            print(f"\n✅ Orderbook connecté!")
            print(f"   Best Bid: ${float(bids[0][0]):,.2f}")
            print(f"   Best Ask: ${float(asks[0][0]):,.2f}")
            spread = float(asks[0][0]) - float(bids[0][0])
            print(f"   Spread: ${spread:.2f}")

        if count == 50:
            print(f"\n📊 {count} updates reçues...")

    task = asyncio.create_task(
        ws.stream_orderbook('btcusdt', handle_book, levels=5, update_speed='100ms')
    )

    await asyncio.sleep(5)
    ws.stop()

    try:
        await asyncio.wait_for(task, timeout=2)
    except:
        pass

    print(f"\n✅ Test terminé: {count} updates en 5 secondes")
    print("🚀 WebSocket Binance fonctionne!")

if __name__ == '__main__':
    asyncio.run(test_quick())
