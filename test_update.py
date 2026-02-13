#!/usr/bin/env python3
"""
Test rapide de l'auto-updater
"""

import os
import sys

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from updater.auto_updater import AutoUpdater


def main():
    print("🧪 Test de l'auto-updater")
    print("=" * 50)

    updater = AutoUpdater()
    print(f"📱 Version actuelle : {updater.current_version}")
    print(f"🌐 URL de vérification : {updater.VERSION_URL}")
    print()

    print("🔍 Vérification des mises à jour...")
    update_info = updater.check_for_updates()

    if update_info:
        print("\n✅ Mise à jour disponible!")
        print(f"   Version : {update_info.get('version')}")
        print(f"   Obligatoire : {update_info.get('mandatory', False)}")
        print(f"   URL Mac : {update_info.get('download_url_mac')}")
        print("\n   Changelog:")
        for item in update_info.get("changelog", []):
            print(f"   • {item}")
    else:
        print("\n✅ Aucune mise à jour disponible (ou erreur)")
        print("   Votre application est à jour!")


if __name__ == "__main__":
    main()
