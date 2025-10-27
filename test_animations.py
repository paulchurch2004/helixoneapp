#!/usr/bin/env python3
"""
Test des nouvelles animations de HelixOne
"""

import sys
sys.path.insert(0, "src")

print("=" * 80)
print("TEST DES ANIMATIONS HELIXONE")
print("=" * 80)
print()

# Test 1: Import des modules d'animation
print("Test 1: Imports des modules d'animation...")
try:
    from src.interface.toast_notifications import ToastManager, ToastNotification
    print("✓ ToastManager et ToastNotification importés")
except Exception as e:
    print(f"✗ Erreur ToastNotifications: {e}")
    sys.exit(1)

try:
    from src.interface.animated_score_widget import AnimatedCircularScore, CompactCircularScore
    print("✓ AnimatedCircularScore et CompactCircularScore importés")
except Exception as e:
    print(f"✗ Erreur AnimatedScoreWidget: {e}")
    sys.exit(1)

try:
    from src.interface.animated_components import (
        AnimatedButton, PageTransition, LoadingSkeleton, AnimatedProgressBar
    )
    print("✓ AnimatedComponents importés")
except Exception as e:
    print(f"✗ Erreur AnimatedComponents: {e}")
    sys.exit(1)

# Test 2: Import de main_app avec les nouvelles dépendances
print("\nTest 2: Import de main_app avec animations...")
try:
    from src.interface import main_app
    print("✓ main_app importé avec succès")
except Exception as e:
    print(f"✗ Erreur import main_app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Vérifier les fonctions critiques
print("\nTest 3: Vérification des fonctions...")
fonctions_requises = [
    'safe_show_notification',
    'display_animated_score',
    'safe_analyser_action',
    'format_rapport_joli'
]

for func in fonctions_requises:
    if hasattr(main_app, func):
        print(f"✓ Fonction '{func}' trouvée")
    else:
        print(f"✗ Fonction '{func}' manquante")
        sys.exit(1)

# Test 4: Vérifier les variables globales
print("\nTest 4: Vérification des variables globales...")
variables_requises = ['toast_manager', 'score_widget_container']

for var in variables_requises:
    if var in dir(main_app):
        print(f"✓ Variable '{var}' trouvée")
    else:
        print(f"✗ Variable '{var}' manquante")
        sys.exit(1)

print()
print("=" * 80)
print("✅ TOUS LES TESTS PASSENT!")
print("=" * 80)
print()
print("🎬 Nouvelles animations ajoutées:")
print()
print("  1. 🎯 Score FXI Circulaire Animé")
print("     - Animation de compteur (0 → score final)")
print("     - Cercle de progression qui se remplit")
print("     - Couleurs dynamiques selon le score")
print("     - Effet glow si score > 75")
print()
print("  2. 🔔 Notifications Toast Animées")
print("     - Slide in/out depuis la droite")
print("     - Icons colorés selon le type")
print("     - Auto-dismiss après 3 secondes")
print("     - Stack de notifications")
print()
print("  3. 🖱️  Boutons avec Hover Animé")
print("     - Scale 1.05x au survol")
print("     - Border glow effect")
print("     - Transitions fluides (0.3s)")
print()
print("  4. 🔄 Transitions de Page Fluides")
print("     - Fade in/out entre sections")
print("     - Slide from right")
print("     - Easing cubic pour smoothness")
print()
print("  5. ⏳ Loading Skeleton")
print("     - Shimmer effect pendant chargement")
print("     - Barres de skeleton avec animation")
print("     - Remplace le \"Loading...\" statique")
print()
print("=" * 80)
print()
print("🚀 Lancez l'application pour voir les animations:")
print("   HELIXONE_DEV=1 python3 run.py")
print()
print("💡 Ce qui va changer:")
print("   • Analysez une action → Score circulaire animé!")
print("   • Notifications élégantes en bas à droite")
print("   • Tous les boutons réagissent au hover")
print("   • Navigation fluide entre les pages")
print()
print("=" * 80)
