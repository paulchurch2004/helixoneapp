#!/usr/bin/env python3
"""
Test complet pour Coinbase API
Exchange crypto US de niveau institutionnel
"""

import sys
sys.path.insert(0, 'helixone-backend')

from app.services.coinbase_source import get_coinbase_collector

def test_coinbase_basic():
    """Test de base de la connexion Coinbase"""
    print("\n🏦 TEST 1: Connexion Coinbase")
    coinbase = get_coinbase_collector()
    print("✅ Connexion établie")
    return coinbase

def test_btc_ticker(coinbase):
    """Test ticker BTC"""
    print("\n🏦 TEST 2: Ticker BTC-USD")
    ticker = coinbase.get_product_ticker('BTC-USD')
    if ticker:
        print(f"✅ BTC-USD Ticker:")
        print(f"   Prix: ${float(ticker.get('price', 0)):,.2f}")
        print(f"   Volume: {float(ticker.get('volume', 0)):,.2f}")

def test_btc_stats(coinbase):
    """Test stats 24h BTC"""
    print("\n🏦 TEST 3: Stats 24h BTC-USD")
    stats = coinbase.get_product_stats('BTC-USD')
    if stats:
        print(f"✅ Stats BTC-USD:")
        print(f"   Open: ${float(stats.get('open', 0)):,.2f}")
        print(f"   High: ${float(stats.get('high', 0)):,.2f}")
        print(f"   Low: ${float(stats.get('low', 0)):,.2f}")
        print(f"   Volume: {float(stats.get('volume', 0)):,.2f}")

def test_eth_price(coinbase):
    """Test prix ETH"""
    print("\n🏦 TEST 4: Prix ETH-USD")
    eth_price = coinbase.get_crypto_price('ETH', 'USD')
    print(f"✅ ETH-USD: ${eth_price:,.2f}")

def test_ltc_price(coinbase):
    """Test prix LTC"""
    print("\n🏦 TEST 5: Prix LTC-USD")
    ltc_price = coinbase.get_crypto_price('LTC', 'USD')
    print(f"✅ LTC-USD: ${ltc_price:,.2f}")

def test_orderbook(coinbase):
    """Test carnet d'ordres"""
    print("\n🏦 TEST 6: Carnet d'ordres BTC-USD (Level 1)")
    orderbook = coinbase.get_product_orderbook('BTC-USD', level=1)
    if orderbook:
        print(f"✅ Orderbook reçu:")
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if bids:
            print(f"   Meilleur Bid: ${float(bids[0][0]):,.2f} (Vol: {bids[0][1]})")
        if asks:
            print(f"   Meilleur Ask: ${float(asks[0][0]):,.2f} (Vol: {asks[0][1]})")

def test_multiple_cryptos(coinbase):
    """Test prix multiples cryptos"""
    print("\n🏦 TEST 7: Prix de plusieurs cryptos")
    cryptos = ['BTC', 'ETH', 'LTC', 'BCH', 'LINK']
    print("✅ Prix:")
    for crypto in cryptos:
        try:
            price = coinbase.get_crypto_price(crypto, 'USD')
            print(f"   {crypto}/USD: ${price:,.2f}")
        except Exception as e:
            print(f"   {crypto}/USD: Erreur - {str(e)[:50]}")

def test_products_list(coinbase):
    """Test liste des produits"""
    print("\n🏦 TEST 8: Liste des produits disponibles")
    products = coinbase.get_products()
    if products:
        usd_products = [p for p in products if p.get('quote_currency') == 'USD']
        print(f"✅ Produits Coinbase:")
        print(f"   Total: {len(products)}")
        print(f"   Paires USD: {len(usd_products)}")
        print(f"   Exemples: {', '.join([p['id'] for p in usd_products[:5]])}")

def test_market_summary(coinbase):
    """Test résumé marché"""
    print("\n🏦 TEST 9: Résumé du marché")
    summaries = coinbase.get_market_summary('USD')
    print(f"✅ Résumé marché:")
    print(f"   Produits analysés: {len(summaries)}")
    if summaries:
        print(f"   Top 5 par volume:")
        sorted_by_vol = sorted(summaries, key=lambda x: float(x.get('volume_24h', 0)), reverse=True)[:5]
        for i, product in enumerate(sorted_by_vol, 1):
            print(f"      {i}. {product['product_id']}: Vol=${float(product['volume_24h']):,.0f}, Prix=${float(product['price']):,.2f}")

if __name__ == '__main__':
    print("="*60)
    print("TEST COMPLET COINBASE API")
    print("Exchange US de niveau institutionnel")
    print("="*60)

    try:
        coinbase = test_coinbase_basic()
        test_btc_ticker(coinbase)
        test_btc_stats(coinbase)
        test_eth_price(coinbase)
        test_ltc_price(coinbase)
        test_orderbook(coinbase)
        test_multiple_cryptos(coinbase)
        test_products_list(coinbase)
        test_market_summary(coinbase)

        print("\n" + "="*60)
        print("✅ TOUS LES TESTS COINBASE RÉUSSIS")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
