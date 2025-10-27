"""
Test des modèles SQLAlchemy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.models import Base, User, License, Analysis
    from app.core.database import engine
    
    print("✅ Imports des modèles réussis")
    print(f"✅ User: {User.__tablename__}")
    print(f"✅ License: {License.__tablename__}")
    print(f"✅ Analysis: {Analysis.__tablename__}")
    
    # Créer les tables
    print("\n🔨 Création des tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées avec succès !")
    
    # Lister les tables créées
    print(f"\n📊 Tables dans la base de données:")
    for table in Base.metadata.sorted_tables:
        print(f"   - {table.name}")
    
    print("\n✅ ✅ ✅ TEST RÉUSSI ✅ ✅ ✅")
    print("\nVous pouvez passer à l'étape suivante !")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()