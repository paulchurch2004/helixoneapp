"""
Application principale HelixOne
Gère la navigation entre login, register et dashboard
"""

import customtkinter as ctk
import sys
import os

# Ajouter le chemin parent pour les imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.auth_manager import AuthManager
from src.interface.login_window import LoginWindow
from src.interface.register_window import RegisterWindow


class HelixOneApp:
    """
    Application principale HelixOne
    Gère l'authentification et la navigation
    """
    
    def __init__(self):
        """Initialiser l'application"""
        
        # Configuration CustomTkinter
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Créer le gestionnaire d'authentification
        self.auth_manager = AuthManager()
        
        # Variable pour stocker la fenêtre actuelle
        self.current_window = None
        
        # Démarrer l'application
        self.start()
    
    def start(self):
        """Démarrer l'application"""
        
        # Vérifier si l'utilisateur est déjà connecté
        if self.auth_manager.is_logged_in():
            print("✅ Session existante détectée")
            user = self.auth_manager.get_current_user()
            license = self.auth_manager.get_license_info()
            print(f"👤 Utilisateur: {user['email']}")
            print(f"📜 Licence: {license['license_type']} - {license['days_remaining']} jours")
            self.show_main_app()
        else:
            print("📋 Aucune session - Affichage login")
            self.show_login()
    
    def show_login(self):
        """Afficher la fenêtre de connexion"""
        
        print("🔵 show_login appelé")
        
        # Fermer la fenêtre actuelle si elle existe
        if self.current_window:
            try:
                self.current_window.destroy()
            except Exception:
                pass
        
        # Créer la fenêtre de login
        self.current_window = LoginWindow(self.auth_manager)
        
        # Définir les callbacks
        self.current_window.set_success_callback(self.on_login_success)
        self.current_window.set_register_callback(self.show_register)
        
        print("🔵 Callbacks définis, lancement mainloop")
        
        # Lancer la boucle
        self.current_window.mainloop()
    
    def show_register(self):
        """Afficher la fenêtre d'inscription"""
        
        print("🟢 show_register appelé!")
        
        # Fermer la fenêtre actuelle
        if self.current_window:
            try:
                print("🟢 Destruction de la fenêtre login")
                self.current_window.quit()  # Quitter mainloop
                self.current_window.destroy()
            except Exception as e:
                print(f"⚠️ Erreur destruction: {e}")
        
        # Créer la fenêtre d'inscription
        print("🟢 Création fenêtre inscription")
        self.current_window = RegisterWindow(self.auth_manager)
        
        # Définir les callbacks
        self.current_window.set_success_callback(self.on_register_success)
        self.current_window.set_login_callback(self.show_login)
        
        print("�� Lancement mainloop inscription")
        
        # Lancer la boucle
        self.current_window.mainloop()
    
    def on_login_success(self):
        """Callback après connexion réussie"""
        print("✅ Connexion réussie - Affichage de l'application principale")
        
        # Récupérer les infos utilisateur
        user = self.auth_manager.get_current_user()
        license = self.auth_manager.get_license_info()
        
        print(f"👤 Utilisateur: {user['email']}")
        print(f"📜 Licence: {license['license_type']} - {license['days_remaining']} jours")
        
        # Quitter le mainloop de login
        if self.current_window:
            self.current_window.quit()
        
        # Afficher l'application principale
        self.show_main_app()
    
    def on_register_success(self):
        """Callback après inscription réussie"""
        print("✅ Inscription réussie - Affichage de l'application principale")
        
        # Récupérer les infos
        user = self.auth_manager.get_current_user()
        license = self.auth_manager.get_license_info()
        
        print(f"👤 Nouvel utilisateur: {user['email']}")
        print(f"📜 Licence d'essai: {license['days_remaining']} jours")
        
        # Quitter le mainloop de register
        if self.current_window:
            self.current_window.quit()
        
        # Afficher l'application principale
        self.show_main_app()
    
    def show_main_app(self):
        """Afficher l'application principale (main_app.py)"""
        
        # Fermer la fenêtre de login/register
        if self.current_window:
            try:
                self.current_window.destroy()
            except Exception:
                pass
        
        # Importer depuis src/interface/main_app.py
        from src.interface import main_app
        
        # Créer la fenêtre principale
        self.current_window = ctk.CTk()
        self.current_window.title("HelixOne - Plateforme de Trading")
        self.current_window.geometry("1400x900")
        
        # Configuration
        self.current_window.configure(fg_color="#0d1117")
        
        # Lancer l'application principale avec auth_manager
        main_app.launch_main_app(
            self.current_window, 
            self.auth_manager, 
            self.on_logout
        )
        
        # Lancer la boucle
        self.current_window.mainloop()
    
    def on_logout(self):
        """Callback lors de la déconnexion"""
        print("🚪 Déconnexion...")
        
        # Supprimer la session
        self.auth_manager.clear_session()
        
        # Fermer l'application principale
        if self.current_window:
            try:
                self.current_window.quit()
                self.current_window.destroy()
            except Exception:
                pass
        
        # Retour au login
        self.show_login()


# Point d'entrée
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 LANCEMENT DE HELIXONE")
    print("=" * 70)
    print()
    
    # Créer et lancer l'application
    app = HelixOneApp()
