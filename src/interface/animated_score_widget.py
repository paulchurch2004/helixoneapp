"""
Widget de Score FXI Circulaire Animé
Affiche un score avec animation de remplissage circulaire
"""

import customtkinter as ctk
from tkinter import Canvas
import math


class AnimatedCircularScore(ctk.CTkFrame):
    """Score FXI avec cercle de progression animé"""

    def __init__(self, parent, size=250, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.size = size
        self.current_value = 0
        self.target_value = 0
        self.animation_running = False

        # Canvas pour le cercle
        self.canvas = Canvas(
            self,
            width=size,
            height=size,
            bg='#0a0e12',
            highlightthickness=0
        )
        self.canvas.pack()

        self.center_x = size // 2
        self.center_y = size // 2
        self.radius = (size - 40) // 2

        # Label pour le score
        self.score_label = ctk.CTkLabel(
            self,
            text="0",
            font=("Segoe UI", 48, "bold"),
            text_color="#ffffff"
        )
        self.score_label.place(relx=0.5, rely=0.45, anchor="center")

        # Label pour "/100"
        self.max_label = ctk.CTkLabel(
            self,
            text="/100",
            font=("Segoe UI", 16),
            text_color="#888888"
        )
        self.max_label.place(relx=0.5, rely=0.58, anchor="center")

        # Label pour le texte
        self.text_label = ctk.CTkLabel(
            self,
            text="Score FXI",
            font=("Segoe UI", 14, "bold"),
            text_color="#00D9FF"
        )
        self.text_label.place(relx=0.5, rely=0.75, anchor="center")

        self._draw_background()

    def _draw_background(self):
        """Dessine le cercle de fond"""
        # Cercle de fond (gris foncé)
        self.canvas.create_oval(
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius,
            outline='#1c2028',
            width=12,
            tags='bg'
        )

    def set_score(self, score: float, label: str = "Score FXI"):
        """Définit et anime le score"""
        self.target_value = max(0, min(100, score))
        self.text_label.configure(text=label)

        if not self.animation_running:
            self.animation_running = True
            self._animate()

    def _animate(self):
        """Anime le score et le cercle"""
        if abs(self.current_value - self.target_value) < 0.5:
            self.current_value = self.target_value
            self._update_display()
            self.animation_running = False
            return

        # Smooth easing
        self.current_value += (self.target_value - self.current_value) * 0.08
        self._update_display()

        # Continue l'animation
        self.after(16, self._animate)  # ~60 FPS

    def _update_display(self):
        """Met à jour l'affichage du score et du cercle"""
        # Mettre à jour le texte du score
        self.score_label.configure(text=f"{int(self.current_value)}")

        # Déterminer la couleur selon le score
        color = self._get_color_for_score(self.current_value)
        self.score_label.configure(text_color=color)

        # Redessiner le cercle
        self._draw_circle()

    def _get_color_for_score(self, score):
        """Retourne la couleur selon le score"""
        if score >= 80:
            return "#00FF88"  # Vert excellent
        elif score >= 65:
            return "#00D9FF"  # Bleu bon
        elif score >= 50:
            return "#FFD700"  # Jaune moyen
        elif score >= 35:
            return "#FFA500"  # Orange faible
        else:
            return "#FF6B6B"  # Rouge mauvais

    def _draw_circle(self):
        """Dessine le cercle de progression"""
        self.canvas.delete('progress')

        # Calculer l'angle (de -90° pour commencer en haut)
        angle = (self.current_value / 100) * 360

        # Couleur du cercle
        color = self._get_color_for_score(self.current_value)

        # Arc de progression
        self.canvas.create_arc(
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius,
            start=90,  # Commence en haut
            extent=-angle,  # Sens horaire
            outline=color,
            width=12,
            style='arc',
            tags='progress'
        )

        # Effet glow si score élevé
        if self.current_value >= 75:
            self._draw_glow(color)

    def _draw_glow(self, color):
        """Ajoute un effet de glow autour du cercle"""
        # Cercle externe légèrement plus grand avec opacité
        glow_radius = self.radius + 8
        self.canvas.create_arc(
            self.center_x - glow_radius,
            self.center_y - glow_radius,
            self.center_x + glow_radius,
            self.center_y + glow_radius,
            start=90,
            extent=-(self.current_value / 100) * 360,
            outline=color,
            width=4,
            style='arc',
            tags='progress'
        )


class CompactCircularScore(ctk.CTkFrame):
    """Version compacte du score circulaire (pour dashboard)"""

    def __init__(self, parent, size=120, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.size = size
        self.current_value = 0
        self.target_value = 0

        self.canvas = Canvas(
            self,
            width=size,
            height=size,
            bg='#0a0e12',
            highlightthickness=0
        )
        self.canvas.pack()

        self.center_x = size // 2
        self.center_y = size // 2
        self.radius = (size - 20) // 2

        self.score_label = ctk.CTkLabel(
            self,
            text="0",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        )
        self.score_label.place(relx=0.5, rely=0.5, anchor="center")

        self._draw_background()

    def _draw_background(self):
        """Dessine le cercle de fond"""
        self.canvas.create_oval(
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius,
            outline='#1c2028',
            width=8,
            tags='bg'
        )

    def set_score(self, score: float):
        """Définit et anime le score"""
        self.target_value = max(0, min(100, score))
        self._animate()

    def _animate(self):
        """Anime le score"""
        if abs(self.current_value - self.target_value) < 0.5:
            self.current_value = self.target_value
            self._update_display()
            return

        self.current_value += (self.target_value - self.current_value) * 0.1
        self._update_display()
        self.after(16, self._animate)

    def _update_display(self):
        """Met à jour l'affichage"""
        self.score_label.configure(text=f"{int(self.current_value)}")

        if self.current_value >= 80:
            color = "#00FF88"
        elif self.current_value >= 65:
            color = "#00D9FF"
        elif self.current_value >= 50:
            color = "#FFD700"
        elif self.current_value >= 35:
            color = "#FFA500"
        else:
            color = "#FF6B6B"

        self.score_label.configure(text_color=color)

        self.canvas.delete('progress')
        angle = (self.current_value / 100) * 360

        self.canvas.create_arc(
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius,
            start=90,
            extent=-angle,
            outline=color,
            width=8,
            style='arc',
            tags='progress'
        )
