"""
Authentification biométrique pour HelixOne
Support de Touch ID/Face ID (macOS) et Windows Hello (Windows)
"""

import platform
from typing import Optional, Callable
import threading


class BiometricAuth:
    """
    Gère l'authentification biométrique
    - macOS: Touch ID / Face ID via LocalAuthentication
    - Windows: Windows Hello via UserConsentVerifier
    - Linux: Non supporté
    """

    def __init__(self):
        """Initialiser l'authentification biométrique"""
        self.platform = platform.system()
        self._context = None
        self._winrt_available = False

        # Vérifier si winrt est disponible sur Windows
        if self.platform == 'Windows':
            try:
                import winrt
                self._winrt_available = True
            except ImportError:
                pass

    def is_available(self) -> bool:
        """
        Vérifier si l'authentification biométrique est disponible

        Returns:
            True si Touch ID/Face ID/Windows Hello est disponible
        """
        if self.platform == 'Darwin':  # macOS
            return self._is_available_macos()
        elif self.platform == 'Windows':
            return self._is_available_windows()
        else:
            return False

    def authenticate(
        self,
        reason: str = "HelixOne souhaite accéder à vos identifiants",
        callback: Optional[Callable[[bool, Optional[str]], None]] = None
    ) -> bool:
        """
        Demander l'authentification biométrique

        Args:
            reason: Message affiché à l'utilisateur
            callback: Fonction appelée avec (success: bool, error: str)

        Returns:
            True si authentification réussie (mode synchrone uniquement)
        """
        if self.platform == 'Darwin':
            if callback:
                # Mode asynchrone (recommandé pour GUI)
                self._authenticate_macos_async(reason, callback)
                return True  # Indication que la requête a été lancée
            else:
                # Mode synchrone
                return self._authenticate_macos_sync(reason)
        elif self.platform == 'Windows':
            if callback:
                # Mode asynchrone (recommandé pour GUI)
                self._authenticate_windows_async(reason, callback)
                return True
            else:
                # Mode synchrone
                return self._authenticate_windows_sync(reason)
        else:
            if callback:
                callback(False, "Biométrie non supportée sur cette plateforme")
            return False

    # ====== macOS Touch ID / Face ID ======

    def _is_available_macos(self) -> bool:
        """Vérifier si Touch ID/Face ID est disponible sur macOS"""
        try:
            from LocalAuthentication import LAContext, LAPolicyDeviceOwnerAuthenticationWithBiometrics

            context = LAContext.alloc().init()
            can_evaluate, error = context.canEvaluatePolicy_error_(
                LAPolicyDeviceOwnerAuthenticationWithBiometrics,
                None
            )

            return can_evaluate

        except ImportError:
            # PyObjC pas installé
            return False
        except Exception as e:
            print(f"Erreur vérification biométrie macOS: {e}")
            return False

    def _authenticate_macos_sync(self, reason: str) -> bool:
        """
        Authentifier avec Touch ID/Face ID (synchrone)

        ATTENTION: Peut bloquer l'interface, préférer la version async
        """
        try:
            from LocalAuthentication import LAContext, LAPolicyDeviceOwnerAuthenticationWithBiometrics
            import objc

            context = LAContext.alloc().init()

            # Configuration du contexte
            context.setLocalizedFallbackTitle_("")  # Masquer "Entrer mot de passe"

            # Variable pour stocker le résultat
            result = {'success': False, 'done': False}

            def completion_handler(success, error):
                result['success'] = success
                result['done'] = True

            # Lancer l'authentification
            context.evaluatePolicy_localizedReason_reply_(
                LAPolicyDeviceOwnerAuthenticationWithBiometrics,
                reason,
                completion_handler
            )

            # Attendre la réponse (timeout 30s)
            import time
            timeout = 30
            elapsed = 0
            while not result['done'] and elapsed < timeout:
                time.sleep(0.1)
                elapsed += 0.1

            return result['success']

        except ImportError:
            print("❌ PyObjC pas installé. Installez: pip install pyobjc-framework-LocalAuthentication")
            return False
        except Exception as e:
            print(f"Erreur authentification biométrique macOS: {e}")
            return False

    def _authenticate_macos_async(self, reason: str, callback: Callable[[bool, Optional[str]], None]):
        """
        Authentifier avec Touch ID/Face ID (asynchrone)

        Args:
            reason: Message affiché
            callback: Fonction appelée avec (success, error_message)
        """
        def run_auth():
            try:
                from LocalAuthentication import LAContext, LAPolicyDeviceOwnerAuthenticationWithBiometrics

                context = LAContext.alloc().init()
                context.setLocalizedFallbackTitle_("")

                def completion_handler(success, error):
                    error_msg = None
                    if error:
                        error_msg = str(error.localizedDescription())
                    callback(success, error_msg)

                # Lancer l'authentification
                context.evaluatePolicy_localizedReason_reply_(
                    LAPolicyDeviceOwnerAuthenticationWithBiometrics,
                    reason,
                    completion_handler
                )

            except ImportError:
                callback(False, "PyObjC pas installé")
            except Exception as e:
                callback(False, f"Erreur: {str(e)}")

        # Lancer dans un thread pour ne pas bloquer l'UI
        thread = threading.Thread(target=run_auth, daemon=True)
        thread.start()

    # ====== Windows Hello ======

    def _is_available_windows(self) -> bool:
        """Vérifier si Windows Hello est disponible"""
        if not self._winrt_available:
            # Essayer sans winrt - simplement vérifier si Windows 10+
            try:
                import sys
                if sys.getwindowsversion().build >= 10240:  # Windows 10 build 10240+
                    return True
            except:
                pass
            return False

        try:
            from winrt.windows.security.credentials.ui import UserConsentVerifier, UserConsentVerifierAvailability

            # Vérifier la disponibilité
            availability = UserConsentVerifier.check_availability_async().get()
            return availability == UserConsentVerifierAvailability.AVAILABLE

        except Exception as e:
            print(f"Erreur vérification Windows Hello: {e}")
            return False

    def _authenticate_windows_sync(self, reason: str) -> bool:
        """Authentifier avec Windows Hello (synchrone)"""
        if not self._winrt_available:
            print("❌ winrt non installé. Installez: pip install winrt-runtime")
            return False

        try:
            from winrt.windows.security.credentials.ui import UserConsentVerifier, UserConsentVerificationResult

            # Demander l'authentification
            result = UserConsentVerifier.request_verification_async(reason).get()

            return result == UserConsentVerificationResult.VERIFIED

        except Exception as e:
            print(f"Erreur authentification Windows Hello: {e}")
            return False

    def _authenticate_windows_async(self, reason: str, callback: Callable[[bool, Optional[str]], None]):
        """Authentifier avec Windows Hello (asynchrone)"""
        def run_auth():
            if not self._winrt_available:
                callback(False, "winrt non installé. Installez: pip install winrt-runtime")
                return

            try:
                from winrt.windows.security.credentials.ui import UserConsentVerifier, UserConsentVerificationResult

                # Demander l'authentification
                result = UserConsentVerifier.request_verification_async(reason).get()

                success = (result == UserConsentVerificationResult.VERIFIED)
                error_msg = None if success else "Authentification refusée ou annulée"

                callback(success, error_msg)

            except Exception as e:
                callback(False, f"Erreur: {str(e)}")

        # Lancer dans un thread pour ne pas bloquer l'UI
        thread = threading.Thread(target=run_auth, daemon=True)
        thread.start()

    # ====== Type de biométrie ======

    def get_biometry_type(self) -> str:
        """
        Obtenir le type de biométrie disponible

        Returns:
            "touchid", "faceid", "windowshello", "none"
        """
        if self.platform == 'Darwin':
            return self._get_biometry_type_macos()
        elif self.platform == 'Windows':
            return "windowshello" if self._is_available_windows() else "none"
        return "none"

    def _get_biometry_type_macos(self) -> str:
        """Obtenir le type de biométrie sur macOS"""
        try:
            from LocalAuthentication import LAContext

            context = LAContext.alloc().init()

            # biometryType disponible depuis macOS 10.13.2
            if hasattr(context, 'biometryType'):
                biometry_type = context.biometryType()

                # 1 = Touch ID, 2 = Face ID
                if biometry_type == 1:
                    return "touchid"
                elif biometry_type == 2:
                    return "faceid"

            return "none"

        except Exception:
            return "none"


# Instance globale
_biometric_auth = BiometricAuth()


def is_biometric_available() -> bool:
    """Raccourci pour vérifier si la biométrie est disponible"""
    return _biometric_auth.is_available()


def authenticate_with_biometric(
    reason: str = "HelixOne souhaite accéder à vos identifiants",
    callback: Optional[Callable[[bool, Optional[str]], None]] = None
) -> bool:
    """Raccourci pour s'authentifier avec biométrie"""
    return _biometric_auth.authenticate(reason, callback)


def get_biometry_type() -> str:
    """Raccourci pour obtenir le type de biométrie"""
    return _biometric_auth.get_biometry_type()


# Test du module
if __name__ == "__main__":
    print("🧪 Test du BiometricAuth\n")

    bio = BiometricAuth()
    print(f"Platform: {bio.platform}")
    print(f"Biométrie disponible: {bio.is_available()}")
    print(f"Type de biométrie: {bio.get_biometry_type()}")

    if bio.is_available():
        print("\n🔐 Test d'authentification biométrique...")

        def on_result(success, error):
            if success:
                print("✅ Authentification réussie!")
            else:
                print(f"❌ Authentification échouée: {error}")

        bio.authenticate(
            reason="Test HelixOne - Authentification biométrique",
            callback=on_result
        )

        # Attendre le résultat
        import time
        time.sleep(5)
    else:
        print("\n⚠️ Biométrie non disponible sur cet appareil")
