"""
Système de Notifications Toast Animées
Messages élégants qui apparaissent en bas à droite
"""

import customtkinter as ctk
from typing import Literal


class ToastNotification(ctk.CTkFrame):
    """Notification toast individuelle"""

    # Couleurs selon le type
    COLORS = {
        "success": {"bg": "#00FF88", "text": "#0a0e12"},
        "error": {"bg": "#FF6B6B", "text": "#ffffff"},
        "warning": {"bg": "#FFA500", "text": "#0a0e12"},
        "info": {"bg": "#00D9FF", "text": "#0a0e12"}
    }

    # Icons selon le type
    ICONS = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ"
    }

    def __init__(self, parent, message: str, notification_type: str = "info", duration: int = 3000):
        colors = self.COLORS.get(notification_type, self.COLORS["info"])

        super().__init__(
            parent,
            fg_color=colors["bg"],
            corner_radius=10,
            border_width=0
        )

        self.duration = duration
        self.slide_distance = 350  # Distance de slide in
        self.current_x = self.slide_distance

        # Container avec padding
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(padx=15, pady=12)

        # Icon
        icon_label = ctk.CTkLabel(
            content_frame,
            text=self.ICONS.get(notification_type, "ℹ"),
            font=("Segoe UI", 18, "bold"),
            text_color=colors["text"]
        )
        icon_label.pack(side="left", padx=(0, 10))

        # Message
        message_label = ctk.CTkLabel(
            content_frame,
            text=message,
            font=("Segoe UI", 13),
            text_color=colors["text"],
            wraplength=250
        )
        message_label.pack(side="left")

        # Bouton fermer
        close_btn = ctk.CTkButton(
            content_frame,
            text="×",
            width=30,
            height=30,
            font=("Segoe UI", 20, "bold"),
            fg_color="transparent",
            text_color=colors["text"],
            hover_color=self._darken_color(colors["bg"]),
            command=self.dismiss
        )
        close_btn.pack(side="left", padx=(10, 0))

    def _darken_color(self, hex_color: str) -> str:
        """Assombrit une couleur hex de 20%"""
        # Convertir hex en RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Assombrir de 20%
        r = int(r * 0.8)
        g = int(g * 0.8)
        b = int(b * 0.8)

        return f"#{r:02x}{g:02x}{b:02x}"

    def show(self):
        """Affiche la notification avec animation slide-in"""
        self._slide_in()

    def _slide_in(self):
        """Animation slide-in depuis la droite"""
        if self.current_x > 0:
            self.current_x -= 20
            self.place(relx=1.0, x=-self.current_x, rely=1.0, y=-20, anchor="se")
            self.after(10, self._slide_in)
        else:
            self.current_x = 0
            self.place(relx=1.0, x=-20, rely=1.0, y=-20, anchor="se")
            # Auto-dismiss après duration
            self.after(self.duration, self.dismiss)

    def dismiss(self):
        """Ferme la notification avec animation slide-out"""
        self._slide_out()

    def _slide_out(self):
        """Animation slide-out vers la droite"""
        if self.current_x < self.slide_distance:
            self.current_x += 20
            self.place(relx=1.0, x=-self.current_x, rely=1.0, y=-20, anchor="se")
            self.after(10, self._slide_out)
        else:
            self.destroy()


class ToastManager:
    """Gestionnaire de notifications toast avec stack"""

    def __init__(self, parent):
        self.parent = parent
        self.toasts = []
        self.max_toasts = 5

    def show(self, message: str, notification_type: Literal["success", "error", "warning", "info"] = "info", duration: int = 3000):
        """Affiche une nouvelle notification"""
        # Limiter le nombre de toasts
        if len(self.toasts) >= self.max_toasts:
            # Retirer le plus ancien
            oldest = self.toasts.pop(0)
            oldest.dismiss()

        # Créer la nouvelle notification
        toast = ToastNotification(self.parent, message, notification_type, duration)
        self.toasts.append(toast)

        # Repositionner toutes les notifications
        self._reposition_toasts()

        # Afficher la nouvelle notification
        toast.show()

        # Nettoyer la liste après destruction
        def cleanup():
            if toast in self.toasts:
                self.toasts.remove(toast)
            self._reposition_toasts()

        toast.bind("<Destroy>", lambda e: cleanup())

    def _reposition_toasts(self):
        """Repositionne toutes les notifications en stack"""
        y_offset = 20
        for i, toast in enumerate(reversed(self.toasts)):
            if toast.winfo_exists():
                toast.place(relx=1.0, x=-20, rely=1.0, y=-(y_offset), anchor="se")
                # Calculer la hauteur du toast pour le prochain
                toast.update_idletasks()
                y_offset += toast.winfo_height() + 10

    def clear_all(self):
        """Ferme toutes les notifications"""
        for toast in self.toasts[:]:
            toast.dismiss()
        self.toasts.clear()
