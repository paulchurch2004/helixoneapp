"""
Module de synchronisation de la progression formation avec le backend
Gère le cache local et la synchronisation cloud
"""

import json
import os
import requests
from typing import Dict, Optional
from datetime import datetime

from src.asset_path import get_base_path
from src.config import get_api_url
from src.auth_session import get_auth_token


class FormationSyncService:
    """Service de synchronisation de la progression formation"""

    def __init__(self, user_email: str):
        self.user_email = user_email
        self.api_url = f"{get_api_url()}/api/formation"
        self._local_cache_path = self._get_cache_path()

    def _get_cache_path(self) -> str:
        """Obtient le chemin du cache local pour cet utilisateur"""
        import hashlib
        user_hash = hashlib.md5(self.user_email.encode()).hexdigest()[:16]
        cache_dir = os.path.join(get_base_path(), "data", "formation_commerciale")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"user_progress_{user_hash}.json")

    def _get_headers(self) -> Dict[str, str]:
        """Obtient les headers d'authentification"""
        token = get_auth_token()
        if token:
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        return {"Content-Type": "application/json"}

    def load_progress(self) -> Dict:
        """
        Charge la progression depuis le backend (avec fallback sur le cache local)

        Returns:
            Dict de progression utilisateur
        """
        # Essayer de charger depuis le backend
        try:
            token = get_auth_token()
            if token:
                response = requests.get(
                    f"{self.api_url}/progress",
                    headers=self._get_headers(),
                    timeout=5
                )

                if response.status_code == 200:
                    backend_data = response.json()
                    # Convertir au format frontend
                    progress = self._convert_backend_to_frontend(backend_data)
                    # Sauvegarder dans le cache local
                    self._save_to_cache(progress)
                    print(f"✅ Progression chargée depuis le backend (XP: {progress['total_xp']})")
                    return progress
        except Exception as e:
            print(f"⚠️ Erreur chargement backend: {e}, utilisation du cache local")

        # Fallback: charger depuis le cache local
        return self._load_from_cache()

    def save_progress(self, progress: Dict) -> bool:
        """
        Sauvegarde la progression localement et synchronise avec le backend

        Args:
            progress: Dictionnaire de progression

        Returns:
            True si sauvegardé avec succès
        """
        # Sauvegarder localement d'abord (toujours)
        success = self._save_to_cache(progress)

        # Essayer de synchroniser avec le backend
        try:
            token = get_auth_token()
            if token:
                # Convertir au format backend
                backend_data = self._convert_frontend_to_backend(progress)

                response = requests.post(
                    f"{self.api_url}/progress/sync",
                    headers=self._get_headers(),
                    json=backend_data,
                    timeout=10
                )

                if response.status_code == 200:
                    print("✅ Progression synchronisée avec le backend")
                    return True
                else:
                    print(f"⚠️ Erreur sync backend: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Erreur sync backend: {e}")

        return success

    def _load_from_cache(self) -> Dict:
        """Charge la progression depuis le cache local"""
        try:
            if os.path.exists(self._local_cache_path):
                with open(self._local_cache_path, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    print(f"📂 Progression chargée depuis le cache local (XP: {progress.get('total_xp', 0)})")
                    return progress
        except Exception as e:
            print(f"⚠️ Erreur lecture cache: {e}")

        # Retourner une progression vide par défaut
        return self._get_default_progress()

    def _save_to_cache(self, progress: Dict) -> bool:
        """Sauvegarde la progression dans le cache local"""
        try:
            with open(self._local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde cache: {e}")
            return False

    def _get_default_progress(self) -> Dict:
        """Retourne une progression par défaut pour un nouvel utilisateur"""
        return {
            "completed_modules": [],
            "quiz_scores": {},
            "total_xp": 0,
            "level": 1,
            "current_parcours": "debutant",
            "badges": [],
            "certifications": [],
            "login_dates": [],
            "community_messages": 0,
            "users_helped": 0
        }

    def _convert_backend_to_frontend(self, backend_data: Dict) -> Dict:
        """
        Convertit les données backend au format frontend

        Le backend utilise: completed_modules (list), module_scores (dict)
        Le frontend utilise: completed_modules (list), quiz_scores (dict)
        """
        progress = self._get_default_progress()

        # Mapper les champs
        progress["total_xp"] = backend_data.get("total_xp", 0)
        progress["level"] = backend_data.get("level", 1)
        progress["completed_modules"] = backend_data.get("completed_modules", [])

        # IMPORTANT: Récupérer les badges et certifications
        if "badges" in backend_data:
            progress["badges"] = backend_data["badges"]
        if "certifications" in backend_data:
            progress["certifications"] = backend_data["certifications"]

        # Convertir module_scores en quiz_scores
        module_scores = backend_data.get("module_scores", {})
        for module_id, score_data in module_scores.items():
            if isinstance(score_data, dict):
                progress["quiz_scores"][module_id] = score_data.get("score", 0)
            else:
                progress["quiz_scores"][module_id] = score_data

        return progress

    def _convert_frontend_to_backend(self, progress: Dict) -> Dict:
        """
        Convertit les données frontend au format backend

        Args:
            progress: Données frontend

        Returns:
            Données au format backend
        """
        # Convertir quiz_scores en module_scores
        module_scores = {}
        for module_id, score in progress.get("quiz_scores", {}).items():
            module_scores[module_id] = {
                "score": score,
                "time_spent": 0,  # TODO: tracker le temps
                "completed_at": datetime.now().isoformat()
            }

        return {
            "total_xp": progress.get("total_xp", 0),
            "level": progress.get("level", 1),
            "completed_modules": progress.get("completed_modules", []),
            "module_scores": module_scores,
            "current_streak": len(progress.get("login_dates", [])),
            "badges": progress.get("badges", []),
            "certifications": progress.get("certifications", [])
        }
