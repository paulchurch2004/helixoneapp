"""
Script de test pour CoinGecko API
Test des cryptos - GRATUIT
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.coingecko_source import get_coingecko_collector


def test_coingecko():
    """Tester CoinGecko API"""

    print("\n" + "="*70)
    print("🪙 TEST COINGECKO API - CRYPTO MARKET DATA")
    print("GRATUIT - 10-50 req/min - Pas de clé requise")
    print("="*70 + "\n")

    cg = get_coingecko_collector()

    # Test 1: Ping
    print("🏓 Test 1: Ping API")
    print("-" * 70)
    try:
        if cg.ping():
            print("✅ API accessible\n")
        else:
            print("❌ API non accessible\n")
            return
    except Exception as e:
        print(f"❌ Erreur: {e}\n")
        return

    # Test 2: Prix simples
    print("💰 Test 2: Prix Bitcoin, Ethereum, Cardano")
    print("-" * 70)
    try:
        prices = cg.get_coin_price(
            ids=['bitcoin', 'ethereum', 'cardano'],
            vs_currencies=['usd', 'eur'],
            include_market_cap=True,
            include_24h_vol=True,
            include_24h_change=True
        )

        for coin_id, data in prices.items():
            print(f"\n📊 {coin_id.upper()}:")
            print(f"   Prix USD: ${data.get('usd', 'N/A'):,.2f}")
            print(f"   Prix EUR: €{data.get('eur', 'N/A'):,.2f}")
            print(f"   Market Cap: ${data.get('usd_market_cap', 0):,.0f}")
            print(f"   Volume 24h: ${data.get('usd_24h_vol', 0):,.0f}")
            print(f"   Change 24h: {data.get('usd_24h_change', 0):.2f}%")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 3: Top 10 cryptos
    print("🏆 Test 3: Top 10 cryptos par market cap")
    print("-" * 70)
    try:
        markets = cg.get_coin_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=10,
            page=1
        )

        print(f"\n{'Rank':<5} {'Nom':<15} {'Prix':<15} {'Market Cap':<20} {'24h %':<10}")
        print("-" * 70)

        for coin in markets:
            rank = coin.get('market_cap_rank', 'N/A')
            name = coin.get('name', 'Unknown')[:14]
            symbol = coin.get('symbol', '').upper()
            price = coin.get('current_price', 0)
            mcap = coin.get('market_cap', 0)
            change = coin.get('price_change_percentage_24h', 0)

            change_symbol = "🔴" if change < 0 else "🟢"

            print(f"{rank:<5} {name:<15} ${price:<14,.2f} ${mcap:<19,.0f} {change_symbol} {change:>6.2f}%")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 4: Historique Bitcoin
    print("📈 Test 4: Historique Bitcoin (7 derniers jours)")
    print("-" * 70)
    try:
        history = cg.get_coin_market_chart(
            coin_id='bitcoin',
            vs_currency='usd',
            days=7
        )

        prices = history.get('prices', [])
        volumes = history.get('total_volumes', [])

        if len(prices) > 0:
            print(f"\n✅ Historique récupéré: {len(prices)} points")

            # Afficher premiers et derniers points
            from datetime import datetime

            first_price = prices[0]
            last_price = prices[-1]

            first_time = datetime.fromtimestamp(first_price[0] / 1000)
            last_time = datetime.fromtimestamp(last_price[0] / 1000)

            print(f"\n   📅 {first_time.strftime('%Y-%m-%d %H:%M')}: ${first_price[1]:,.2f}")
            print(f"   📅 {last_time.strftime('%Y-%m-%d %H:%M')}: ${last_price[1]:,.2f}")

            change_pct = ((last_price[1] - first_price[1]) / first_price[1]) * 100
            change_symbol = "📈" if change_pct > 0 else "📉"
            print(f"\n   {change_symbol} Variation 7j: {change_pct:+.2f}%")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 5: Données globales
    print("🌍 Test 5: Données globales marché crypto")
    print("-" * 70)
    try:
        global_data = cg.get_global_data()

        total_mcap = global_data.get('total_market_cap', {}).get('usd', 0)
        total_vol = global_data.get('total_volume', {}).get('usd', 0)
        btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
        eth_dominance = global_data.get('market_cap_percentage', {}).get('eth', 0)
        active_cryptos = global_data.get('active_cryptocurrencies', 0)
        markets = global_data.get('markets', 0)

        print(f"\n   💰 Market Cap Total: ${total_mcap:,.0f}")
        print(f"   📊 Volume 24h Total: ${total_vol:,.0f}")
        print(f"   🪙 Bitcoin Dominance: {btc_dominance:.2f}%")
        print(f"   💎 Ethereum Dominance: {eth_dominance:.2f}%")
        print(f"   🔢 Cryptos actives: {active_cryptos:,}")
        print(f"   🏦 Marchés: {markets:,}")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 6: Trending
    print("🔥 Test 6: Cryptos tendances (Top 7)")
    print("-" * 70)
    try:
        trending = cg.get_trending()

        coins = trending.get('coins', [])

        if coins:
            print()
            for i, item in enumerate(coins, 1):
                coin = item.get('item', {})
                name = coin.get('name', 'Unknown')
                symbol = coin.get('symbol', '').upper()
                rank = coin.get('market_cap_rank', 'N/A')
                score = coin.get('score', 0)

                print(f"   {i}. {name} ({symbol}) - Rank: #{rank} - Score: {score}")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 7: Recherche
    print("🔍 Test 7: Recherche 'doge'")
    print("-" * 70)
    try:
        results = cg.search_coins('doge')

        coins = results.get('coins', [])

        if coins:
            print(f"\n✅ {len(coins)} résultats trouvés:\n")

            for coin in coins[:5]:  # Top 5 résultats
                name = coin.get('name', 'Unknown')
                symbol = coin.get('symbol', '').upper()
                rank = coin.get('market_cap_rank', 'N/A')

                print(f"   • {name} ({symbol}) - Rank: #{rank}")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST COINGECKO")
    print("="*70)
    print("✅ Tous les tests exécutés")
    print("🪙 CoinGecko API est maintenant intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT (10-50 req/min)")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ 13,000+ cryptos disponibles")
    print("  - ✅ Prix, market cap, volumes")
    print("  - ✅ Historique illimité")
    print("  - ✅ Données globales marché")
    print("  - ✅ Trending, catégories, exchanges")
    print("\nDonnées disponibles:")
    print("  - 💰 Prix en temps réel (multi-devises)")
    print("  - 📈 Historique complet (jusqu'à 'max')")
    print("  - 🌍 Données globales marché crypto")
    print("  - 🔥 Cryptos tendances")
    print("  - 🏦 Exchanges et volumes")
    print("  - 📂 Catégories et classifications")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_coingecko()
