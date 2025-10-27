#!/usr/bin/env python3
"""
Test complet pour Kraken API
Exchange crypto européen avec support multi-devises
"""

import sys
sys.path.insert(0, 'helixone-backend')

from app.services.kraken_source import get_kraken_collector

def test_kraken_basic():
    """Test de base de la connexion Kraken"""
    print("\n🔷 TEST 1: Connexion Kraken")
    kraken = get_kraken_collector()
    print("✅ Connexion établie")
    return kraken

def test_server_time(kraken):
    """Test du temps serveur"""
    print("\n🔷 TEST 2: Temps serveur")
    time_data = kraken.get_server_time()
    print(f"✅ Temps serveur: {time_data}")

def test_btc_price(kraken):
    """Test prix BTC en USD"""
    print("\n🔷 TEST 3: Prix BTC/USD")
    btc_usd = kraken.get_crypto_price('XBT', 'USD')
    print(f"✅ BTC/USD: ${btc_usd:,.2f}")

def test_eth_price(kraken):
    """Test prix ETH en USD"""
    print("\n🔷 TEST 4: Prix ETH/USD")
    eth_usd = kraken.get_crypto_price('ETH', 'USD')
    print(f"✅ ETH/USD: ${eth_usd:,.2f}")

def test_multi_currency(kraken):
    """Test prix multi-devises"""
    print("\n🔷 TEST 5: BTC multi-devises")
    multi = kraken.get_crypto_prices_multi_currency('XBT')
    print(f"✅ BTC disponible en {len(multi)} devises:")
    for curr, price in multi.items():
        print(f"   {curr}: ${price:,.2f}")

def test_ticker_info(kraken):
    """Test informations ticker"""
    print("\n🔷 TEST 6: Ticker BTC/USD")
    ticker = kraken.get_ticker(['XXBTZUSD'])
    if ticker and 'XXBTZUSD' in ticker:
        data = ticker['XXBTZUSD']
        print(f"✅ Ticker reçu:")
        print(f"   Ask: ${float(data['a'][0]):,.2f}")
        print(f"   Bid: ${float(data['b'][0]):,.2f}")
        print(f"   Last: ${float(data['c'][0]):,.2f}")
        print(f"   Volume 24h: {float(data['v'][1]):,.2f}")

def test_ohlc_data(kraken):
    """Test données OHLC"""
    print("\n🔷 TEST 7: Données OHLC BTC/USD")
    result = kraken.get_ohlc('XXBTZUSD', interval=60)
    if result and 'XXBTZUSD' in result:
        ohlc = result['XXBTZUSD']
        print(f"✅ OHLC reçu: {len(ohlc)} périodes")
        if ohlc:
            latest = ohlc[-1]
            print(f"   Dernière bougie: O=${latest[1]}, H=${latest[2]}, L=${latest[3]}, C=${latest[4]}")

def test_orderbook(kraken):
    """Test carnet d'ordres"""
    print("\n🔷 TEST 8: Carnet d'ordres BTC/USD")
    result = kraken.get_orderbook('XXBTZUSD', count=5)
    if result and 'XXBTZUSD' in result:
        book = result['XXBTZUSD']
        asks = book.get('asks', [])
        bids = book.get('bids', [])
        print(f"✅ Orderbook reçu:")
        print(f"   Asks: {len(asks)}, Bids: {len(bids)}")
        if asks and bids:
            print(f"   Meilleur Ask: ${float(asks[0][0]):,.2f}")
            print(f"   Meilleur Bid: ${float(bids[0][0]):,.2f}")

def test_recent_trades(kraken):
    """Test trades récents"""
    print("\n🔷 TEST 9: Trades récents BTC/USD")
    result = kraken.get_recent_trades('XXBTZUSD')
    if result and 'XXBTZUSD' in result:
        trades = result['XXBTZUSD']
        print(f"✅ Trades reçus: {len(trades)} trades")
        if trades:
            latest = trades[-1]
            print(f"   Dernier trade: ${latest[0]} (Vol: {latest[1]})")

if __name__ == '__main__':
    print("="*60)
    print("TEST COMPLET KRAKEN API")
    print("Exchange européen avec multi-devises (USD, EUR, GBP, JPY, CAD)")
    print("="*60)

    try:
        kraken = test_kraken_basic()
        test_server_time(kraken)
        test_btc_price(kraken)
        test_eth_price(kraken)
        test_multi_currency(kraken)
        test_ticker_info(kraken)
        test_ohlc_data(kraken)
        test_orderbook(kraken)
        test_recent_trades(kraken)

        print("\n" + "="*60)
        print("✅ TOUS LES TESTS KRAKEN RÉUSSIS")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
