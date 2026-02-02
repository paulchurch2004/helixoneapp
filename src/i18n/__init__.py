"""
Système d'internationalisation (i18n) pour HelixOne
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

class TranslationManager:
    """Gestionnaire de traductions multilingues"""

    _instance = None
    _current_language = "fr"  # Langue par défaut
    _translations: Dict[str, Any] = {}
    _available_languages = ["fr", "en"]

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(TranslationManager, cls).__new__(cls)
            cls._instance._load_translations()
        return cls._instance

    def _load_translations(self):
        """Charge les traductions depuis le fichier JSON"""
        try:
            translations_file = Path(__file__).parent / "translations.json"
            with open(translations_file, 'r', encoding='utf-8') as f:
                self._translations = json.load(f)
            print(f"✅ Traductions chargées : {list(self._translations.keys())}")
        except Exception as e:
            print(f"❌ Erreur chargement traductions : {e}")
            self._translations = {"fr": {}, "en": {}}

    def set_language(self, lang_code: str):
        """
        Définit la langue active

        Args:
            lang_code: Code de la langue (fr, en, etc.)
        """
        if lang_code in self._available_languages:
            self._current_language = lang_code
            print(f"🌍 Langue changée : {lang_code.upper()}")
            return True
        else:
            print(f"⚠️ Langue non disponible : {lang_code}")
            return False

    def get_language(self) -> str:
        """Retourne la langue active"""
        return self._current_language

    def get_available_languages(self) -> list:
        """Retourne la liste des langues disponibles"""
        return self._available_languages.copy()

    def translate(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        Traduit une clé dans la langue active

        Args:
            key: Clé de traduction (ex: 'app.title', 'auth.login')
            default: Valeur par défaut si la clé n'existe pas
            **kwargs: Variables à interpoler dans la traduction

        Returns:
            Texte traduit

        Examples:
            >>> t = TranslationManager()
            >>> t.translate('app.title')
            'HelixOne - Analyse d\'actions avec IA'
            >>> t.translate('app.loading')
            'Chargement...'
        """
        keys = key.split('.')
        current = self._translations.get(self._current_language, {})

        # Navigation dans l'arbre de traduction
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
            else:
                current = None
                break

        # Si la traduction n'existe pas, retourner la clé ou le défaut
        if current is None:
            result = default if default is not None else key
        else:
            result = str(current)

        # Interpolation des variables
        if kwargs:
            try:
                result = result.format(**kwargs)
            except KeyError:
                pass

        return result

    def t(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """Alias court pour translate()"""
        return self.translate(key, default, **kwargs)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Retourne toutes les traductions d'une section

        Args:
            section: Nom de la section (ex: 'auth', 'menu', 'settings')

        Returns:
            Dictionnaire des traductions de la section
        """
        return self._translations.get(self._current_language, {}).get(section, {})


# Instance globale du gestionnaire de traductions
_translator = TranslationManager()

# Fonctions d'accès global
def t(key: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Fonction globale de traduction

    Usage:
        from src.i18n import t

        label = ctk.CTkLabel(text=t('auth.login'))
        button = ctk.CTkButton(text=t('app.confirm'))
    """
    return _translator.translate(key, default, **kwargs)

def set_language(lang_code: str) -> bool:
    """Définit la langue globale de l'application"""
    return _translator.set_language(lang_code)

def get_language() -> str:
    """Retourne la langue active"""
    return _translator.get_language()

def get_available_languages() -> list:
    """Retourne les langues disponibles"""
    return _translator.get_available_languages()

def get_section(section: str) -> Dict[str, Any]:
    """Retourne toutes les traductions d'une section"""
    return _translator.get_section(section)


# Classe pour sauvegarder/charger les préférences de langue
class LanguagePreferences:
    """Gère la persistance des préférences de langue"""

    PREF_FILE = Path.home() / ".helixone" / "language.json"

    @classmethod
    def save_language(cls, lang_code: str):
        """Sauvegarde la langue préférée"""
        try:
            cls.PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(cls.PREF_FILE, 'w', encoding='utf-8') as f:
                json.dump({"language": lang_code}, f)
            return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde langue : {e}")
            return False

    @classmethod
    def load_language(cls) -> Optional[str]:
        """Charge la langue préférée"""
        try:
            if cls.PREF_FILE.exists():
                with open(cls.PREF_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("language")
        except Exception as e:
            print(f"❌ Erreur chargement langue : {e}")
        return None

    @classmethod
    def initialize_language(cls):
        """Initialise la langue au démarrage de l'application"""
        saved_lang = cls.load_language()
        if saved_lang:
            set_language(saved_lang)
            print(f"🌍 Langue chargée depuis les préférences : {saved_lang.upper()}")
        else:
            print(f"🌍 Utilisation de la langue par défaut : FR")


# Initialiser la langue au chargement du module
LanguagePreferences.initialize_language()


__all__ = [
    't',
    'set_language',
    'get_language',
    'get_available_languages',
    'get_section',
    'TranslationManager',
    'LanguagePreferences'
]
