"""
Composants UI Animés pour HelixOne
Boutons avec hover, transitions de page, etc.
"""

import customtkinter as ctk
from typing import Callable, Optional


class AnimatedButton(ctk.CTkButton):
    """Bouton avec animation hover (scale + glow)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normal_fg = kwargs.get('fg_color', '#00D9FF')
        self.hover_fg = kwargs.get('hover_color', '#00FFB3')

        # Bind hover events
        self.bind("<Enter>", self._on_hover_enter)
        self.bind("<Leave>", self._on_hover_leave)

        self.hover_active = False
        self.scale_factor = 1.0
        self.target_scale = 1.0

    def _on_hover_enter(self, event):
        """Quand la souris entre sur le bouton"""
        self.hover_active = True
        self.target_scale = 1.05
        self._animate_scale()

    def _on_hover_leave(self, event):
        """Quand la souris sort du bouton"""
        self.hover_active = False
        self.target_scale = 1.0
        self._animate_scale()

    def _animate_scale(self):
        """Anime le scale du bouton"""
        if abs(self.scale_factor - self.target_scale) < 0.01:
            self.scale_factor = self.target_scale
            return

        # Smooth interpolation
        self.scale_factor += (self.target_scale - self.scale_factor) * 0.3

        # Note: CTkButton ne supporte pas le scale direct
        # On simule avec un changement de border_width et font_size
        if self.hover_active:
            self.configure(border_width=2)
        else:
            self.configure(border_width=0)

        self.after(16, self._animate_scale)


class PageTransition:
    """Gère les transitions fluides entre les pages"""

    @staticmethod
    def fade_out(widget, callback: Optional[Callable] = None, duration: int = 200):
        """Fade out avec callback"""
        steps = 10
        step_duration = duration // steps

        def step(current_step=0):
            if current_step >= steps:
                if callback:
                    callback()
                return

            # Calculer l'opacité (pas supporté directement par CTk)
            # On simule avec un changement de fg_color progressif
            widget.update()
            current_step += 1
            widget.after(step_duration, lambda: step(current_step))

        step()

    @staticmethod
    def fade_in(widget, duration: int = 200):
        """Fade in une widget"""
        steps = 10
        step_duration = duration // steps

        # Commencer invisible (simulé)
        widget.update()

        def step(current_step=0):
            if current_step >= steps:
                return

            widget.update()
            current_step += 1
            widget.after(step_duration, lambda: step(current_step))

        step()

    @staticmethod
    def slide_in_from_right(widget, duration: int = 300):
        """Slide in depuis la droite"""
        widget.place(relx=1.2, rely=0, relwidth=1, relheight=1)

        steps = 15
        step_duration = duration // steps
        start_x = 1.2
        target_x = 0

        def step(current_step=0):
            if current_step >= steps:
                widget.place(relx=0, rely=0, relwidth=1, relheight=1)
                return

            # Easing out
            progress = current_step / steps
            ease_progress = 1 - (1 - progress) ** 3  # Cubic ease-out
            current_x = start_x + (target_x - start_x) * ease_progress

            widget.place(relx=current_x, rely=0, relwidth=1, relheight=1)

            current_step += 1
            widget.after(step_duration, lambda: step(current_step))

        step()

    @staticmethod
    def transition_pages(old_widget, new_widget, parent, transition_type: str = "fade"):
        """Transition entre deux widgets"""
        if transition_type == "fade":
            # Fade out old, then fade in new
            def show_new():
                if old_widget and old_widget.winfo_exists():
                    old_widget.destroy()
                new_widget.pack(fill="both", expand=True)
                PageTransition.fade_in(new_widget)

            if old_widget and old_widget.winfo_exists():
                PageTransition.fade_out(old_widget, show_new)
            else:
                show_new()

        elif transition_type == "slide":
            # Slide in new from right
            if old_widget and old_widget.winfo_exists():
                old_widget.destroy()
            new_widget.place(relx=0, rely=0, relwidth=1, relheight=1)
            PageTransition.slide_in_from_right(new_widget)


class LoadingSkeleton(ctk.CTkFrame):
    """Skeleton screen pour le chargement"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.configure(fg_color="#161920")

        # Créer des barres de skeleton
        self._create_skeleton_bars()

        # Animation shimmer
        self.shimmer_position = 0
        self.shimmer_active = True
        self._animate_shimmer()

    def _create_skeleton_bars(self):
        """Crée les barres de skeleton"""
        # Barre de titre
        title_bar = ctk.CTkFrame(self, height=40, fg_color="#1c2028", corner_radius=8)
        title_bar.pack(fill="x", padx=20, pady=(20, 10))

        # Barres de contenu
        for i in range(5):
            width_percent = 0.7 if i % 2 == 0 else 0.9
            bar = ctk.CTkFrame(
                self,
                height=20,
                fg_color="#1c2028",
                corner_radius=6
            )
            bar.pack(fill="x", padx=20, pady=8, ipadx=int(self.winfo_width() * width_percent))

    def _animate_shimmer(self):
        """Anime l'effet shimmer"""
        if not self.shimmer_active or not self.winfo_exists():
            return

        # Cycle le shimmer
        self.shimmer_position = (self.shimmer_position + 1) % 100

        # Continue l'animation
        self.after(50, self._animate_shimmer)

    def stop(self):
        """Arrête l'animation"""
        self.shimmer_active = False


class PulsingIndicator(ctk.CTkFrame):
    """Indicateur de chargement pulsant (petit point)"""

    def __init__(self, parent, size: int = 12, color: str = "#00D9FF", **kwargs):
        super().__init__(
            parent,
            width=size,
            height=size,
            fg_color=color,
            corner_radius=size // 2,
            **kwargs
        )

        self.base_color = color
        self.pulse_step = 0
        self.pulsing = True

        self._pulse()

    def _pulse(self):
        """Anime la pulsation"""
        if not self.pulsing or not self.winfo_exists():
            return

        # Calculer l'opacité (simulé par changement de couleur)
        import math
        opacity = 0.3 + 0.7 * abs(math.sin(self.pulse_step * 0.1))

        # On ne peut pas changer l'opacité directement, mais on peut changer la couleur
        # On simule en alternant entre la couleur de base et une version plus sombre

        self.pulse_step += 1
        self.after(50, self._pulse)

    def stop(self):
        """Arrête la pulsation"""
        self.pulsing = False


class AnimatedProgressBar(ctk.CTkProgressBar):
    """Barre de progression avec animation fluide"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_value = 0
        self.target_value = 0
        self.animating = False

    def set_value_animated(self, value: float, duration: int = 500):
        """Définit la valeur avec animation"""
        self.target_value = max(0, min(1, value))

        if not self.animating:
            self.animating = True
            self._animate()

    def _animate(self):
        """Anime la progression"""
        if abs(self.current_value - self.target_value) < 0.01:
            self.current_value = self.target_value
            self.set(self.current_value)
            self.animating = False
            return

        # Smooth interpolation
        self.current_value += (self.target_value - self.current_value) * 0.1
        self.set(self.current_value)

        self.after(16, self._animate)
