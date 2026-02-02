"""
Toast Notification System - Notifications non-intrusives avec animations
Version améliorée avec:
- Animations d'entrée/sortie fluides
- Barre de progression
- Niveau "success" pour les confirmations
- Boutons d'action optionnels
"""
import customtkinter as ctk
from tkinter import messagebox
import threading
import time
from typing import Literal, Callable, Optional


class ToastNotification(ctk.CTkToplevel):
    """
    Notification toast non-intrusive avec animations fluides
    """

    def __init__(
        self,
        parent,
        message: str,
        level: Literal["info", "success", "warning", "error", "critical"] = "info",
        duration: int = 5000,
        action_text: str = None,
        action_callback: Callable = None,
        show_progress: bool = True
    ):
        super().__init__(parent)

        self.message = message
        self.level = level
        self.duration = duration
        self.action_text = action_text
        self.action_callback = action_callback
        self.show_progress = show_progress
        self.progress = 0
        self._closing = False

        self.setup_window()
        self.create_widgets()
        self.position_toast()
        self.animate_in()

    def setup_window(self):
        """Configure la fenêtre toast"""
        # Retirer les décorations de fenêtre
        self.overrideredirect(True)

        # Toujours au-dessus
        self.attributes('-topmost', True)

        # Transparence initiale pour animation
        try:
            self.attributes('-alpha', 0.0)
        except Exception:
            pass

        # Taille
        self.geometry("420x100")

    def create_widgets(self):
        """Crée les widgets de la notification avec barre de progression"""
        # Couleurs selon le niveau (ajout de success)
        color_map = {
            "info": ("#1a4a6e", "#00D9FF", "#ffffff"),
            "success": ("#1a4e3a", "#00FF88", "#ffffff"),
            "warning": ("#4e4a1a", "#FFD700", "#ffffff"),
            "error": ("#4e1a1a", "#FF4444", "#ffffff"),
            "critical": ("#6e1a1a", "#FF0000", "#ffffff")
        }

        bg_color, accent_color, text_color = color_map.get(
            self.level, ("#1a4a6e", "#00D9FF", "#ffffff")
        )

        # Frame principal avec bordure accent
        self.main_frame = ctk.CTkFrame(
            self,
            fg_color=bg_color,
            corner_radius=12,
            border_width=1,
            border_color=accent_color
        )
        self.main_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Barre de progression en haut
        if self.show_progress and self.duration > 0:
            self.progress_bar = ctk.CTkFrame(
                self.main_frame,
                height=3,
                corner_radius=0,
                fg_color=accent_color
            )
            self.progress_bar.place(x=0, y=0, relwidth=1)

        # Container du contenu
        content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=15, pady=(10, 12))

        # Icône selon le niveau
        icon_map = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }
        icon = icon_map.get(self.level, "ℹ️")

        icon_label = ctk.CTkLabel(
            content_frame,
            text=icon,
            font=("Arial", 24),
            text_color=text_color
        )
        icon_label.pack(side="left", padx=(0, 12))

        # Container texte
        text_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True)

        # Titre selon le niveau
        title_map = {
            "info": "Information",
            "success": "Succès",
            "warning": "Attention",
            "error": "Erreur",
            "critical": "ALERTE CRITIQUE"
        }
        title = title_map.get(self.level, "Notification")

        title_label = ctk.CTkLabel(
            text_frame,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=accent_color,
            anchor="w"
        )
        title_label.pack(fill="x")

        # Message
        message_label = ctk.CTkLabel(
            text_frame,
            text=self.message,
            font=ctk.CTkFont(size=12),
            text_color=text_color,
            wraplength=280,
            justify="left",
            anchor="w"
        )
        message_label.pack(fill="x", pady=(2, 0))

        # Container boutons (à droite)
        btn_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=(10, 0))

        # Bouton action (optionnel)
        if self.action_text and self.action_callback:
            action_btn = ctk.CTkButton(
                btn_frame,
                text=self.action_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=accent_color,
                hover_color=self._lighten_color(accent_color),
                text_color="#000000",
                width=70, height=28,
                corner_radius=6,
                command=self._on_action
            )
            action_btn.pack(pady=(0, 5))

        # Bouton fermer
        close_btn = ctk.CTkButton(
            btn_frame,
            text="✕",
            width=28, height=28,
            fg_color="transparent",
            hover_color="#ffffff20",
            text_color="#AAAAAA",
            command=self.close_toast
        )
        close_btn.pack()

    def _lighten_color(self, hex_color: str) -> str:
        """Éclaircir une couleur"""
        hex_color = hex_color.lstrip('#')
        r = min(255, int(int(hex_color[0:2], 16) * 1.2))
        g = min(255, int(int(hex_color[2:4], 16) * 1.2))
        b = min(255, int(int(hex_color[4:6], 16) * 1.2))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_action(self):
        """Exécute l'action et ferme le toast"""
        if self.action_callback:
            try:
                self.action_callback()
            except Exception as e:
                print(f"Erreur action toast: {e}")
        self.close_toast()

    def position_toast(self):
        """Positionne le toast en haut à droite de l'écran (hors écran pour animation)"""
        self.update_idletasks()

        # Dimensions de l'écran
        screen_width = self.winfo_screenwidth()

        # Dimensions du toast
        toast_width = self.winfo_width() or 420
        toast_height = self.winfo_height() or 100

        # Position initiale: hors écran à droite
        self.target_x = screen_width - toast_width - 20
        self.start_x = screen_width + 10  # Hors écran
        self.target_y = 80

        self.geometry(f"{toast_width}x{toast_height}+{self.start_x}+{self.target_y}")

    def animate_in(self):
        """Animation d'entrée (slide from right + fade in)"""
        steps = 15
        x_diff = self.start_x - self.target_x
        step_x = x_diff / steps

        def do_step(current_step):
            if current_step > steps or self._closing:
                # Animation terminée, démarrer la progression
                if self.duration > 0:
                    self._start_progress()
                return

            try:
                # Position
                current_x = int(self.start_x - (step_x * current_step))
                self.geometry(f"+{current_x}+{self.target_y}")

                # Opacité
                alpha = min(0.95, current_step / steps)
                self.attributes('-alpha', alpha)

                self.after(15, lambda: do_step(current_step + 1))
            except Exception:
                pass

        do_step(0)

    def _start_progress(self):
        """Démarre la barre de progression"""
        if not hasattr(self, 'progress_bar') or self._closing:
            return

        interval = 50  # ms
        total_steps = self.duration / interval
        step_width = 1 / total_steps

        def update_progress():
            if self._closing:
                return

            self.progress += step_width

            if self.progress >= 1:
                self.close_toast()
                return

            try:
                remaining_width = 1 - self.progress
                self.progress_bar.place(x=0, y=0, relwidth=remaining_width)
                self.after(interval, update_progress)
            except Exception:
                pass

        self.after(interval, update_progress)

    def close_toast(self):
        """Ferme le toast avec animation de sortie"""
        if self._closing:
            return
        self._closing = True

        def do_close():
            steps = 10
            screen_width = self.winfo_screenwidth()
            start_x = self.target_x
            end_x = screen_width + 10
            step_x = (end_x - start_x) / steps

            def animate_out(current_step):
                if current_step > steps:
                    try:
                        self.destroy()
                    except Exception:
                        pass
                    return

                try:
                    # Slide out
                    current_x = int(start_x + (step_x * current_step))
                    self.geometry(f"+{current_x}+{self.target_y}")

                    # Fade out
                    alpha = max(0, 0.95 - (current_step / steps) * 0.95)
                    self.attributes('-alpha', alpha)

                    self.after(12, lambda: animate_out(current_step + 1))
                except Exception:
                    try:
                        self.destroy()
                    except Exception:
                        pass

            animate_out(0)

        # Exécuter sur le thread principal
        try:
            self.after(0, do_close)
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass


class ToastManager:
    """
    Gestionnaire de notifications toast amélioré
    - Empile les notifications si plusieurs sont affichées
    - Support des actions et callbacks
    - Niveau "success" pour les confirmations
    """

    def __init__(self, parent):
        self.parent = parent
        self.active_toasts = []
        self.toast_spacing = 110  # Espacement entre les toasts

    def show_toast(
        self,
        message: str,
        level: Literal["info", "success", "warning", "error", "critical"] = "info",
        duration: int = 5000,
        action_text: str = None,
        action_callback: Callable = None,
        show_progress: bool = True
    ):
        """
        Affiche une notification toast

        Args:
            message: Message à afficher
            level: Niveau d'alerte (info, success, warning, error, critical)
            duration: Durée d'affichage en millisecondes (0 = infini)
            action_text: Texte du bouton d'action (optionnel)
            action_callback: Callback du bouton d'action (optionnel)
            show_progress: Afficher la barre de progression
        """
        try:
            toast = ToastNotification(
                self.parent, message, level, duration,
                action_text, action_callback, show_progress
            )

            # Ajouter à la liste
            self.active_toasts.append(toast)

            # Positionner le toast en tenant compte des autres toasts actifs
            self.reposition_toasts()

            # Retirer de la liste quand détruit
            original_close = toast.close_toast

            def on_close():
                original_close()
                if toast in self.active_toasts:
                    self.active_toasts.remove(toast)
                self.after_safe(100, self.reposition_toasts)

            toast.close_toast = on_close

        except Exception as e:
            print(f"Erreur création toast: {e}")

    def after_safe(self, delay: int, callback: Callable):
        """Appelle after de manière sécurisée"""
        try:
            self.parent.after(delay, callback)
        except Exception:
            pass

    def show_success(self, message: str, duration: int = 4000):
        """Raccourci pour afficher un toast de succès"""
        self.show_toast(message, "success", duration)

    def show_error(self, message: str, duration: int = 6000):
        """Raccourci pour afficher un toast d'erreur"""
        self.show_toast(message, "error", duration)

    def show_warning(self, message: str, duration: int = 5000):
        """Raccourci pour afficher un toast d'avertissement"""
        self.show_toast(message, "warning", duration)

    def show_info(self, message: str, duration: int = 4000):
        """Raccourci pour afficher un toast d'information"""
        self.show_toast(message, "info", duration)

    def reposition_toasts(self):
        """Repositionne tous les toasts actifs"""
        try:
            screen_width = self.parent.winfo_screenwidth()

            for i, toast in enumerate(self.active_toasts):
                if toast.winfo_exists():
                    toast_width = toast.winfo_width()
                    toast_height = toast.winfo_height()

                    x = screen_width - toast_width - 20
                    y = 80 + (i * self.toast_spacing)

                    toast.geometry(f"{toast_width}x{toast_height}+{x}+{y}")
        except Exception:
            pass

    def show_market_alert(self, alert_data: dict):
        """
        Affiche une alerte de marché sous forme de toast

        Args:
            alert_data: Dictionnaire contenant level, type, message
        """
        alert_level = alert_data.get('level', {}).get('name', 'MEDIUM')
        alert_type = alert_data.get('type', 'MARKET')
        alert_message = alert_data.get('message', 'Alerte marché')

        # Mapper les niveaux d'alerte aux niveaux de toast
        level_map = {
            'NONE': 'info',
            'LOW': 'info',
            'MEDIUM': 'warning',
            'HIGH': 'error',
            'CRITICAL': 'critical'
        }

        toast_level = level_map.get(alert_level, 'warning')

        # N'afficher que les alertes HIGH et CRITICAL par défaut
        if alert_level in ['HIGH', 'CRITICAL']:
            self.show_toast(
                f"[{alert_type}] {alert_message}",
                level=toast_level,
                duration=10000 if alert_level == 'CRITICAL' else 7000
            )

    def clear_all(self):
        """Ferme toutes les notifications actives"""
        for toast in self.active_toasts[:]:
            try:
                toast.destroy()
            except Exception:
                pass
        self.active_toasts.clear()


# Fonction d'aide pour utilisation simple
def show_toast(
    parent,
    message: str,
    level: Literal["info", "warning", "error", "critical"] = "info",
    duration: int = 5000
):
    """
    Affiche une notification toast simple

    Args:
        parent: Widget parent (fenêtre principale)
        message: Message à afficher
        level: Niveau d'alerte (info, warning, error, critical)
        duration: Durée d'affichage en millisecondes
    """
    ToastNotification(parent, message, level, duration)


# Test standalone
if __name__ == "__main__":
    app = ctk.CTk()
    app.title("Test Toast Notifications")
    app.geometry("800x600")
    app.configure(fg_color="#1a1d24")

    manager = ToastManager(app)

    def test_info():
        manager.show_info("Ceci est une notification d'information")

    def test_success():
        manager.show_success("Opération réussie avec succès !")

    def test_warning():
        manager.show_warning("Attention : VIX en hausse de 15%")

    def test_error():
        manager.show_error("Erreur : Connexion au serveur perdue")

    def test_critical():
        manager.show_toast(
            "ALERTE : Sell-off détecté ! 80% des actions en baisse",
            "critical",
            duration=10000
        )

    def test_with_action():
        def on_action():
            print("Action exécutée !")
        manager.show_toast(
            "Nouvelle mise à jour disponible",
            "info",
            duration=8000,
            action_text="Installer",
            action_callback=on_action
        )

    def test_multiple():
        manager.show_info("Notification 1")
        app.after(300, lambda: manager.show_success("Notification 2"))
        app.after(600, lambda: manager.show_warning("Notification 3"))

    def test_market_alert():
        alert = {
            'level': {'name': 'CRITICAL'},
            'type': 'MARKET_SELL_OFF',
            'message': 'Sell-off majeur détecté: 85% des actions en baisse'
        }
        manager.show_market_alert(alert)

    # Interface de test
    ctk.CTkLabel(
        app,
        text="Test des Notifications Toast",
        font=ctk.CTkFont(size=24, weight="bold"),
        text_color="#00D9FF"
    ).pack(pady=30)

    btn_frame = ctk.CTkFrame(app, fg_color="transparent")
    btn_frame.pack(pady=20)

    buttons = [
        ("Info", test_info, "#00D9FF"),
        ("Succès", test_success, "#00FF88"),
        ("Avertissement", test_warning, "#FFD700"),
        ("Erreur", test_error, "#FF4444"),
        ("Critique", test_critical, "#FF0000"),
        ("Avec Action", test_with_action, "#9966FF"),
        ("Multiple", test_multiple, "#00BFFF"),
    ]

    for text, cmd, color in buttons:
        ctk.CTkButton(
            btn_frame,
            text=text,
            command=cmd,
            fg_color=color,
            hover_color=color,
            text_color="#000000" if color in ["#FFD700", "#00FF88"] else "#FFFFFF",
            width=100
        ).pack(side="left", padx=5)

    ctk.CTkButton(
        app,
        text="Fermer toutes les notifications",
        command=manager.clear_all,
        fg_color="#333333",
        hover_color="#444444"
    ).pack(pady=20)

    app.mainloop()
