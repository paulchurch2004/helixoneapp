#!/usr/bin/env python3
"""Test de connexion IBKR simplifié"""

from ib_insync import IB
import time

print("\n" + "="*60)
print("🔌 TEST CONNEXION IBKR - MODE LIVE")
print("="*60)

ib = IB()

print("\n📡 Tentative de connexion...")
print("   Host: 127.0.0.1")
print("   Port: 7496 🔴 (LIVE)")
print("   Client ID: 1")
print()

try:
    ib.connect('127.0.0.1', 7496, clientId=1, timeout=10)

    print("✅ CONNEXION RÉUSSIE!\n")

    # Infos compte
    accounts = ib.managedAccounts()
    print(f"📊 Comptes: {accounts}")

    if accounts:
        account = accounts[0]
        print(f"   Compte actif: {account}\n")

        # Attendre un peu pour recevoir les données
        print("⏳ Récupération des données...")
        ib.sleep(2)

        # Résumé compte
        print("\n💰 RÉSUMÉ DU COMPTE:")
        print("-" * 60)

        summary = ib.accountSummary(account)
        for item in summary:
            if item.tag in ['NetLiquidation', 'TotalCashValue', 'GrossPositionValue']:
                print(f"   {item.tag:25s}: {item.value:>15s} {item.currency}")

        # Positions
        print("\n📦 POSITIONS:")
        print("-" * 60)
        positions = ib.positions()

        if positions:
            print(f"   Total: {len(positions)} positions\n")
            for pos in positions:
                symbol = pos.contract.symbol
                qty = pos.position
                cost = pos.avgCost
                print(f"   • {symbol:10s} {qty:>10.2f} @ {cost:>10.2f}")
        else:
            print("   Aucune position ouverte")

        print("\n" + "="*60)
        print("✅ TEST RÉUSSI - INTÉGRATION IBKR FONCTIONNELLE!")
        print("="*60 + "\n")

    ib.disconnect()

except ConnectionRefusedError:
    print("❌ Connexion refusée\n")
    print("🔧 Vérifications:")
    print("   1. TWS est-il lancé?")
    print("   2. Es-tu connecté à ton compte LIVE?")
    print("   3. API activée? (File > Global Config > API > Settings)")
    print("   4. Port 7496 configuré?")
    print("   5. 'Enable ActiveX and Socket Clients' coché?\n")

except TimeoutError:
    print("❌ Timeout - TWS ne répond pas\n")
    print("🔧 TWS est peut-être en train de démarrer...")
    print("   Attends 30 secondes et relance le script\n")

except Exception as e:
    print(f"❌ Erreur: {type(e).__name__}")
    print(f"   Message: {str(e)}\n")
