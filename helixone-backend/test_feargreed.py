"""
Script de test pour Crypto Fear & Greed Index
Test du sentiment crypto - GRATUIT et ILLIMITÉ
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.feargreed_source import get_feargreed_collector


def test_feargreed():
    """Tester Crypto Fear & Greed Index API"""

    print("\n" + "="*70)
    print("😨 TEST CRYPTO FEAR & GREED INDEX")
    print("GRATUIT - ILLIMITÉ - Pas de clé API requise")
    print("="*70 + "\n")

    fg = get_feargreed_collector()

    # Test 1: Current Index
    print("📊 Test 1: Indice Actuel")
    print("-" * 70)
    try:
        current = fg.get_current()

        value = int(current['value'])
        classification = current['value_classification']
        timestamp = datetime.fromtimestamp(int(current['timestamp']))

        # Emoji based on value
        if value <= 24:
            emoji = "😱"
        elif value <= 49:
            emoji = "😨"
        elif value <= 74:
            emoji = "😊"
        else:
            emoji = "🤑"

        print(f"\n{emoji} Indice Fear & Greed: {value}/100")
        print(f"   Classification: {classification}")
        print(f"   Dernière mise à jour: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Visual bar
        bar_length = 50
        filled = int((value / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n   0 {bar} 100")
        print(f"   😱 Fear                          Greed 🤑\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 2: Detailed Interpretation
    print("📖 Test 2: Interprétation Détaillée")
    print("-" * 70)
    try:
        interpretation = fg.get_index_with_interpretation()

        print(f"\n💡 Valeur: {interpretation['value']}/100 ({interpretation['classification']})")
        print(f"\n📝 Interprétation:")
        print(f"   {interpretation['interpretation']}")
        print(f"\n💰 Conseil Trading:")
        print(f"   {interpretation['trading_advice']}\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 3: Historical Data (7 days)
    print("📈 Test 3: Historique 7 Jours")
    print("-" * 70)
    try:
        history = fg.get_historical(limit=7)

        if history:
            print(f"\n✅ {len(history)} points historiques récupérés\n")

            print(f"{'Date':<12} {'Valeur':<8} {'Classification':<20}")
            print("-" * 70)

            for point in history:
                date = datetime.fromtimestamp(int(point['timestamp']))
                value = point['value']
                classification = point['value_classification']

                date_str = date.strftime('%Y-%m-%d')

                print(f"{date_str:<12} {value:<8} {classification:<20}")

            print()

        else:
            print("⚠️  Aucune donnée historique\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 4: Trend Analysis
    print("📊 Test 4: Analyse de Tendance (7 jours)")
    print("-" * 70)
    try:
        trend = fg.get_trend(days=7)

        print(f"\n📈 Tendance sur {trend['days_analyzed']} jours:\n")
        print(f"   Valeur actuelle:  {trend['current_value']}")
        print(f"   Valeur précédente: {trend['previous_value']}")
        print(f"   Changement:       {trend['change']:+d} ({trend['change_percent']:+.2f}%)")
        print(f"   Moyenne période:  {trend['average']:.2f}")
        print(f"   Tendance:         {trend['trend']}")

        # Arrow based on trend
        if trend['change'] > 0:
            print(f"   Direction:        📈 Vers le Greed")
        elif trend['change'] < 0:
            print(f"   Direction:        📉 Vers le Fear")
        else:
            print(f"   Direction:        ↔️ Stable")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 5: Extreme Sentiment Detection
    print("⚠️  Test 5: Détection Sentiment Extrême")
    print("-" * 70)
    try:
        extreme = fg.is_extreme_sentiment(threshold_fear=25, threshold_greed=75)

        print(f"\n🔍 Analyse du sentiment:\n")
        print(f"   Sentiment extrême: {'Oui' if extreme['is_extreme'] else 'Non'}")
        print(f"   Type: {extreme['type'].upper()}")
        print(f"   Message: {extreme['message']}")

        if extreme['is_extreme']:
            print(f"\n   ⚠️  ALERTE: Sentiment extrême détecté!")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 6: Statistics (30 days)
    print("📊 Test 6: Statistiques 30 Jours")
    print("-" * 70)
    try:
        stats = fg.get_statistics(days=30)

        print(f"\n📊 Statistiques sur {stats['total_days']} jours:\n")
        print(f"   Min:         {stats['min']}")
        print(f"   Max:         {stats['max']}")
        print(f"   Moyenne:     {stats['average']:.2f}")
        print(f"   Médiane:     {stats['median']}")
        print(f"   Écart-type:  {stats['std_dev']:.2f}")
        print(f"\n   Jours en Fear:  {stats['days_in_fear']} ({stats['days_in_fear']/stats['total_days']*100:.1f}%)")
        print(f"   Jours en Greed: {stats['days_in_greed']} ({stats['days_in_greed']/stats['total_days']*100:.1f}%)")

        # Dominant sentiment
        if stats['days_in_fear'] > stats['days_in_greed']:
            print(f"\n   💡 Sentiment dominant: FEAR 😨")
        elif stats['days_in_greed'] > stats['days_in_fear']:
            print(f"\n   💡 Sentiment dominant: GREED 😊")
        else:
            print(f"\n   💡 Sentiment dominant: ÉQUILIBRÉ ⚖️")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST FEAR & GREED INDEX")
    print("="*70)
    print("✅ Tous les tests exécutés")
    print("😨 Crypto Fear & Greed Index intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT et ILLIMITÉ")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ Mis à jour toutes les 8 heures")
    print("  - ✅ Historique complet disponible")
    print("  - ✅ Échelle 0-100 (Fear → Greed)")
    print("\nDonnées disponibles:")
    print("  - 📊 Indice actuel avec classification")
    print("  - 📈 Historique illimité")
    print("  - 📉 Analyse de tendance")
    print("  - ⚠️  Détection sentiments extrêmes")
    print("  - 📊 Statistiques sur période")
    print("  - 💡 Interprétation et conseils trading")
    print("\nUtilisation:")
    print("  - Analyse sentiment marché crypto")
    print("  - Détection opportunités achat/vente")
    print("  - Confirmation stratégies contrariennes")
    print("  - Indicateur complémentaire analyses techniques")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_feargreed()
