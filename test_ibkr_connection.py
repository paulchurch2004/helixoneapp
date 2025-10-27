#!/usr/bin/env python3
"""
Script de test pour la connexion Interactive Brokers
"""

from ib_insync import IB, util
import time

def test_connection():
    """Tester la connexion à IB Gateway"""

    print("=" * 60)
    print("🔌 TEST DE CONNEXION INTERACTIVE BROKERS")
    print("=" * 60)
    print()

    # Créer instance IB
    ib = IB()

    # Paramètres de connexion
    # Port 7497 = Paper Trading (simulation)
    # Port 7496 = Live Trading (réel)
    HOST = '127.0.0.1'
    PORT = 7496  # 🔴 LIVE TRADING (compte réel)
    CLIENT_ID = 1

    print(f"📡 Tentative de connexion à IB Gateway...")
    print(f"   Host: {HOST}")
    print(f"   Port: {PORT} 🔴 (LIVE TRADING - COMPTE RÉEL)")
    print(f"   Client ID: {CLIENT_ID}")
    print()
    print("   ⚠️  MODE LIVE: Connexion à ton vrai compte IBKR")
    print("   ⚠️  Read-only: Lecture du portefeuille uniquement")
    print()

    try:
        # Tentative de connexion
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)

        print("✅ CONNEXION RÉUSSIE!")
        print()

        # Récupérer les informations du compte
        print("📊 INFORMATIONS DU COMPTE:")
        print("-" * 60)

        accounts = ib.managedAccounts()
        print(f"   Comptes disponibles: {accounts}")
        print()

        # Récupérer quelques infos du compte
        if accounts:
            account = accounts[0]
            print(f"   Compte actif: {account}")

            # Récupérer le résumé du compte
            print("\n   Récupération des données du compte...")
            time.sleep(2)  # Attendre un peu pour que les données arrivent

            account_values = ib.accountSummary(account)

            if account_values:
                print(f"\n   📈 RÉSUMÉ DU COMPTE:")
                print(f"   " + "-" * 56)

                # Afficher les valeurs importantes
                important_tags = [
                    'NetLiquidation',
                    'TotalCashValue',
                    'GrossPositionValue',
                    'UnrealizedPnL',
                    'RealizedPnL',
                    'AvailableFunds',
                    'BuyingPower'
                ]

                for tag in important_tags:
                    for av in account_values:
                        if av.tag == tag:
                            print(f"   {av.tag:25s}: {av.value:>15s} {av.currency}")
                            break

            # Récupérer les positions
            print("\n   📦 POSITIONS:")
            print(f"   " + "-" * 56)

            positions = ib.positions()

            if positions:
                print(f"   Nombre de positions: {len(positions)}")
                for pos in positions:
                    print(f"   • {pos.contract.symbol:10s} {pos.position:>10.2f} @ {pos.avgCost:>10.2f}")
            else:
                print("   Aucune position ouverte")

            # Récupérer les ordres récents
            print("\n   📝 ORDRES RÉCENTS:")
            print(f"   " + "-" * 56)

            trades = ib.trades()

            if trades:
                print(f"   Nombre d'ordres: {len(trades)}")
                for trade in trades[:5]:  # Afficher les 5 derniers
                    print(f"   • {trade.contract.symbol:10s} {trade.order.action:4s} {trade.order.totalQuantity:>6.0f} @ {trade.order.orderType}")
            else:
                print("   Aucun ordre récent")

        print()
        print("=" * 60)
        print("✅ TEST TERMINÉ AVEC SUCCÈS")
        print("=" * 60)
        print()
        print("💡 PROCHAINES ÉTAPES:")
        print("   1. L'intégration IBKR est fonctionnelle ✅")
        print("   2. Tu peux maintenant implémenter les fonctionnalités:")
        print("      - Monitoring temps réel du portefeuille")
        print("      - Alertes automatiques")
        print("      - Analyse avec le moteur de scénarios")
        print()

        # Déconnecter proprement
        ib.disconnect()
        return True

    except ConnectionRefusedError:
        print("❌ ERREUR: Connexion refusée")
        print()
        print("🔧 SOLUTIONS:")
        print("   1. Assure-toi que IB Gateway est lancé")
        print("   2. Vérifie que l'API est activée dans IB Gateway:")
        print("      Configuration > API > Settings > Enable ActiveX and Socket Clients")
        print("   3. Vérifie le port:")
        print("      - Paper Trading: Port 7497")
        print("      - Live Trading: Port 7496")
        print("   4. Redémarre IB Gateway si nécessaire")
        print()
        return False

    except TimeoutError:
        print("❌ ERREUR: Timeout de connexion")
        print()
        print("🔧 SOLUTIONS:")
        print("   1. IB Gateway est peut-être bloqué")
        print("   2. Redémarre IB Gateway")
        print("   3. Vérifie ton firewall")
        print()
        return False

    except Exception as e:
        print(f"❌ ERREUR INATTENDUE: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print()
        print("🔧 SOLUTION:")
        print("   Vérifie que IB Gateway est correctement installé et configuré")
        print()
        return False


def check_ib_gateway_running():
    """Vérifier si IB Gateway est en cours d'exécution"""
    import subprocess

    print("🔍 Vérification si IB Gateway est lancé...")
    print()

    try:
        # Chercher le processus Java de IB Gateway
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        if 'ibgateway' in result.stdout.lower() or 'tws' in result.stdout.lower():
            print("✅ IB Gateway/TWS détecté en cours d'exécution")
            print()
            return True
        else:
            print("⚠️  IB Gateway/TWS ne semble pas être lancé")
            print()
            print("📝 INSTRUCTIONS POUR LANCER IB GATEWAY:")
            print("   1. Ouvre 'Applications' sur ton Mac")
            print("   2. Trouve 'IB Gateway' ou 'Trader Workstation'")
            print("   3. Double-clique pour lancer")
            print("   4. Connecte-toi avec tes identifiants IBKR")
            print("   5. Va dans Configuration > API Settings")
            print("   6. Coche 'Enable ActiveX and Socket Clients'")
            print("   7. Vérifie que le port est 7497 (paper) ou 7496 (live)")
            print()

            response = input("IB Gateway est-il maintenant lancé? (o/n): ")
            return response.lower() == 'o'

    except Exception as e:
        print(f"⚠️  Impossible de vérifier: {e}")
        print()
        return False


if __name__ == "__main__":
    print()

    # Vérifier si IB Gateway est lancé
    if not check_ib_gateway_running():
        print("⚠️  Lance IB Gateway et relance ce script")
        print()
        exit(1)

    # Tester la connexion
    success = test_connection()

    if success:
        print("🎉 Tout fonctionne! Tu es prêt pour l'intégration complète.")
    else:
        print("⚠️  Résous les erreurs ci-dessus et réessaye.")

    print()
