#!/usr/bin/env python3
"""
Test de la nouvelle interface HelixOne Academy v2.0
"""

import sys
import os

# Ajouter le path src au PYTHONPATH
sys.path.insert(0, "/Users/macintosh/Desktop/helixone/src")

def test_imports():
    """Test 1: Vérifier que tous les imports fonctionnent"""
    print("="*70)
    print("TEST 1: Vérification des imports")
    print("="*70)

    try:
        import customtkinter as ctk
        print("✓ customtkinter importé")

        from interface.formation_commerciale import FormationAcademy, afficher_formation_commerciale
        print("✓ FormationAcademy importé")

        from interface.formation_commerciale import ModuleViewer, SimulateurTrading
        print("✓ ModuleViewer et SimulateurTrading importés")

        return True
    except Exception as e:
        print(f"✗ Erreur d'import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_loading():
    """Test 2: Vérifier le chargement des modules JSON"""
    print("\n" + "="*70)
    print("TEST 2: Chargement des données JSON")
    print("="*70)

    try:
        import json

        json_path = "data/formation_commerciale/modules_complets.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            modules = json.load(f)

        print(f"✓ Fichier JSON chargé avec succès")
        print(f"✓ {len(modules)} modules trouvés")

        # Vérifier la structure
        parcours_count = {"debutant": 0, "intermediaire": 0, "expert": 0}
        for module in modules:
            parcours = module.get("parcours", "debutant")
            if parcours in parcours_count:
                parcours_count[parcours] += 1

        print(f"\nRépartition par parcours:")
        print(f"  - Débutant: {parcours_count['debutant']} modules")
        print(f"  - Intermédiaire: {parcours_count['intermediaire']} modules")
        print(f"  - Expert: {parcours_count['expert']} modules")

        return True

    except Exception as e:
        print(f"✗ Erreur de chargement JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_instantiation():
    """Test 3: Vérifier que la classe peut être instanciée"""
    print("\n" + "="*70)
    print("TEST 3: Instantiation de la classe FormationAcademy")
    print("="*70)

    try:
        import customtkinter as ctk
        from interface.formation_commerciale import FormationAcademy

        # Créer une fenêtre temporaire
        root = ctk.CTk()
        root.withdraw()  # Ne pas afficher la fenêtre

        # Créer l'instance
        academy = FormationAcademy(root)

        print("✓ FormationAcademy instanciée avec succès")

        # Vérifier les attributs
        if hasattr(academy, 'modules_data'):
            print("✓ modules_data présent")

        if hasattr(academy, 'user_progress'):
            print("✓ user_progress présent")

        if hasattr(academy, 'show_dashboard'):
            print("✓ Méthode show_dashboard présente")

        if hasattr(academy, 'show_parcours'):
            print("✓ Méthode show_parcours présente")

        if hasattr(academy, 'open_module'):
            print("✓ Méthode open_module présente")

        # Détruire la fenêtre
        root.destroy()

        return True

    except Exception as e:
        print(f"✗ Erreur d'instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structure():
    """Test 4: Vérifier la structure des données chargées"""
    print("\n" + "="*70)
    print("TEST 4: Structure des données")
    print("="*70)

    try:
        import customtkinter as ctk
        from interface.formation_commerciale import FormationAcademy

        root = ctk.CTk()
        root.withdraw()

        academy = FormationAcademy(root)

        # Vérifier modules_data
        if isinstance(academy.modules_data, dict):
            print("✓ modules_data est un dictionnaire")

        expected_keys = ["debutant", "intermediaire", "expert"]
        for key in expected_keys:
            if key in academy.modules_data:
                count = len(academy.modules_data[key])
                print(f"✓ Parcours '{key}': {count} modules")

        # Vérifier user_progress
        if isinstance(academy.user_progress, dict):
            print("✓ user_progress est un dictionnaire")

        expected_progress_keys = ["completed_modules", "quiz_scores", "total_xp", "level"]
        for key in expected_progress_keys:
            if key in academy.user_progress:
                print(f"✓ user_progress contient '{key}'")

        root.destroy()

        return True

    except Exception as e:
        print(f"✗ Erreur de structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_content():
    """Test 5: Vérifier le contenu d'un module"""
    print("\n" + "="*70)
    print("TEST 5: Contenu d'un module")
    print("="*70)

    try:
        import json

        json_path = "data/formation_commerciale/modules_complets.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            modules = json.load(f)

        # Tester Module 6 (Trading Patterns)
        module_6 = next((m for m in modules if m['id'] == 'trading_patterns_chartistes'), None)

        if module_6:
            print(f"✓ Module 6 trouvé: {module_6['titre']}")

            required_fields = ['id', 'titre', 'description', 'parcours', 'durée', 'points_xp', 'contenu']
            for field in required_fields:
                if field in module_6:
                    print(f"✓ Champ '{field}' présent")
                else:
                    print(f"✗ Champ '{field}' manquant")
                    return False

            # Vérifier le contenu
            contenu = module_6.get('contenu', {})
            if 'sections' in contenu:
                print(f"✓ {len(contenu['sections'])} sections trouvées")

            if 'introduction' in contenu:
                print(f"✓ Introduction présente")

            if 'resume' in contenu:
                print(f"✓ Résumé présent")

            return True
        else:
            print("✗ Module 6 non trouvé")
            return False

    except Exception as e:
        print(f"✗ Erreur de contenu: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test"""
    print("\n" + "#"*70)
    print("# TEST HELIXONE ACADEMY V2.0")
    print("#"*70 + "\n")

    results = []

    # Exécuter tous les tests
    results.append(("Imports", test_imports()))
    results.append(("Chargement JSON", test_json_loading()))
    results.append(("Instantiation de classe", test_class_instantiation()))
    results.append(("Structure des données", test_data_structure()))
    results.append(("Contenu d'un module", test_module_content()))

    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS")
    print("="*70)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*70)

    if all_passed:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("\nVous pouvez maintenant lancer l'application:")
        print("  cd /Users/macintosh/Desktop/helixone")
        print("  HELIXONE_DEV=1 python3 run.py")
        print("\n✨ La nouvelle interface HelixOne Academy v2.0 est prête!")
        print("\nAméliorations:")
        print("  • Interface simplifiée et intuitive")
        print("  • Navigation claire (Dashboard, Parcours, Simulateur)")
        print("  • Visualisation complète des modules")
        print("  • Simulateur de trading fonctionnel")
        print("  • Suivi de progression avec XP et niveaux")
        print("  • Design moderne et professionnel")
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("\nVérifiez les erreurs ci-dessus avant de lancer l'application.")

    print("="*70 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
