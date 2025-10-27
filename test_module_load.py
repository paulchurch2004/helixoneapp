#!/usr/bin/env python3
# Test rapide pour vérifier le chargement des modules

import json
import sys
import os

# Ajouter le path src au PYTHONPATH
sys.path.insert(0, "/Users/macintosh/Desktop/helixone/src")

def test_json_loading():
    """Test 1: Charger le JSON directement"""
    print("="*60)
    print("TEST 1: Chargement du fichier JSON")
    print("="*60)

    try:
        with open("data/formation_commerciale/modules_complets.json", 'r', encoding='utf-8') as f:
            modules = json.load(f)

        print(f"✓ Fichier JSON chargé avec succès")
        print(f"✓ Nombre de modules: {len(modules)}")

        for module in modules:
            print(f"  - {module.get('id')}: {module.get('titre')} ({module.get('parcours')})")

        return True
    except Exception as e:
        print(f"✗ Erreur lors du chargement du JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_data_structure():
    """Test 2: Vérifier la structure des données"""
    print("\n" + "="*60)
    print("TEST 2: Structure des données des modules")
    print("="*60)

    try:
        with open("data/formation_commerciale/modules_complets.json", 'r', encoding='utf-8') as f:
            modules = json.load(f)

        # Test Module 6 (Trading Patterns)
        module_6 = next((m for m in modules if m['id'] == 'trading_patterns_chartistes'), None)

        if module_6:
            print(f"✓ Module 6 trouvé: {module_6['titre']}")
            print(f"  - Parcours: {module_6['parcours']}")
            print(f"  - Durée: {module_6['durée']}")
            print(f"  - Points XP: {module_6['points_xp']}")
            print(f"  - Nombre de sections: {len(module_6['contenu']['sections'])}")
            print(f"  - Nombre de questions quiz: {len(module_6['quiz'])}")
            print(f"  - Nombre d'exercices: {len(module_6['exercices'])}")
        else:
            print("✗ Module 6 non trouvé")
            return False

        # Test Module 7 (Trading Strategies)
        module_7 = next((m for m in modules if m['id'] == 'trading_strategies_advanced'), None)

        if module_7:
            print(f"\n✓ Module 7 trouvé: {module_7['titre']}")
            print(f"  - Parcours: {module_7['parcours']}")
            print(f"  - Durée: {module_7['durée']}")
            print(f"  - Points XP: {module_7['points_xp']}")
            print(f"  - Nombre de sections: {len(module_7['contenu']['sections'])}")
        else:
            print("✗ Module 7 non trouvé")
            return False

        return True

    except Exception as e:
        print(f"✗ Erreur lors de la vérification: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_formation_interface_loading():
    """Test 3: Importer la classe FormationCommerciale"""
    print("\n" + "="*60)
    print("TEST 3: Import de la classe FormationCommerciale")
    print("="*60)

    try:
        from interface.formation_commerciale import FormationCommerciale
        print("✓ Classe FormationCommerciale importée avec succès")
        return True
    except Exception as e:
        print(f"✗ Erreur lors de l'import: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "#"*60)
    print("# TEST DES MODULES DE FORMATION HELIXONE")
    print("#"*60 + "\n")

    results = []

    # Test 1
    results.append(("Chargement JSON", test_json_loading()))

    # Test 2
    results.append(("Structure des données", test_module_data_structure()))

    # Test 3
    results.append(("Import FormationCommerciale", test_formation_interface_loading()))

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 Tous les tests sont passés!")
        print("\nVous pouvez maintenant lancer l'application avec:")
        print("  cd /Users/macintosh/Desktop/helixone && HELIXONE_DEV=1 python3 run.py")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
